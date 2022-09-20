from arch.Arch import Arch
from memopt.graph import Node, find_topo_sort
from memopt.bestfit import BestFit
from .utils import TileInfo
from .config import Config

import functools
import numpy as np
from queue import PriorityQueue
import math
import tvm
from typing import Generator, List, Dict
from .common import coalesced_factor, factorize, get_all_factors, coalesced_tensor_shape

def score_block_size(n):
    num_wrap = (n + 31) // 32
    r1 = max(num_wrap/4, 4/num_wrap)
    r2 = (num_wrap * 32 - n) / n
    return (r1, r2)

def get_block_size(n):
    factors = get_all_factors(n)
    factors = list(filter(lambda x: x <= 1024, factors))
    factor_ordered = sorted(factors, key=score_block_size)
    return factor_ordered[0]

class DefaultPolicy:
    def __init__(self, output_nodes: List[Node], arch:Arch) -> None:
        self.arch = arch
        self.ordered_nodes = list(filter(
            lambda n: not n.is_placeholder() and not n.is_output(),
            find_topo_sort(output_nodes)
        ))
        self.output_nodes = output_nodes

    def emit_config(self, topk: int) -> List[Dict[Node, Config]]:
        base_tile = self.get_base_tile()
        if base_tile is None:
            return []
        rstep_map = {node : self._assign_reduce_step(node) for node in self.ordered_nodes}
        # print(rstep_map)
        smem_tile_condidates = self.DFS_smem_tile(base_tile, topk, rstep_map)
        results = []
        for tile, info in smem_tile_condidates:
            final_rstep_map = self._expand_reduce_axis(tile, info, rstep_map)
            if not self.check_tile_shape_isvalid(tile):
                continue
            # info = self.compute_smem_tile_meta_data(self.get_tile_map(tile), final_rstep_map)
            # print(tile, final_rstep_map, info.smem_cost, info.num_wave, info.block_per_SM)
            block_orders = self._assign_block_order(tile)
            for codegen_dicts in self.assign_block_size(tile, final_rstep_map):
                # handle cases where block is not ordinal (e.g. transpose)
                for node, block_order in block_orders.items():
                    codegen_dicts[node].block_order = block_order
                results.append(codegen_dicts)
                if len(results) >= topk:break
            if len(results) >= topk:break
        return results

    def DFS_smem_tile(self, init_tile, topk, rstep_map):
        _steps = [get_all_factors(n) for n in self.output_nodes[0].get_shape()]
        steps = [step[step.index(t):] for step, t in zip(_steps, init_tile)]
        for i in range(len(steps)):
            added = list(filter(lambda s:s < steps[i][-1] and s > steps[i][0] and s not in steps[i], [2, 4, 8, 16, 32]))
            steps[i].extend(added)
            steps[i] = sorted(steps[i])
        visited_tile = {}
        queue = PriorityQueue()
        def prio(info):
            return (info.footprint + 1) * info.num_wave # * (info.block_per_SM ** 0.5)
        def add_to_queue(tile):
            if tuple(tile) in visited_tile:
                return
            tile_map = self.get_tile_map(tile)
            info = self.compute_smem_tile_meta_data(tile_map, rstep_map)
            visited_tile[tuple(tile)] = info
            if info.valid():
                queue.put([prio(info), tile])

        add_to_queue(init_tile)
        while not (queue.empty() or len(visited_tile) > 2000):
            _, tile = queue.get()
            dim_ids = [step.index(t) for step, t in zip(steps, tile)]
            for i in reversed(range(len(dim_ids))):
                if dim_ids[i] + 1 < len(steps[i]):
                    new_tile = tile.copy()
                    new_tile[i] = steps[i][dim_ids[i] + 1]
                    add_to_queue(new_tile)

        visited_tile = {k : v for k, v in visited_tile.items() if v.valid()}
        sorted_tiles = sorted(visited_tile.items(), key=lambda x:prio(x[1]))
        return sorted_tiles

    # get the minimum tile that could satisfy no redundancy computation
    def get_base_tile(self):
        if len(set([len(node.get_shape()) for node in self.output_nodes])) > 1:
            # If output dim sizes are not same, don't know how to handle them
            return None
        out_node = self.output_nodes[0]
        shape = out_node.get_shape()
        base_tile = [1 for _ in shape]
        wpi = self.compute_workload_per_item(base_tile)
        for dim, n in enumerate(shape):
            # factors = get_all_factors(n)
            factors = [n]
            for factor in factors:
                if factor == base_tile[dim]:continue
                tile = base_tile.copy()
                tile[dim] = factor
                new_wpi = self.compute_workload_per_item(tile)
                if new_wpi < wpi:
                    wpi, base_tile = new_wpi, tile
                else:
                    break

        if self._check_basic_tile(base_tile):
            return None

        return base_tile

    # get_tile_map handles multiple output cases
    def get_tile_map(self, tile):
        tile_map = {}
        for node in self.output_nodes:
            tile_map[node] = [
                tile[i] * node.get_shape()[i] // self.output_nodes[0].get_shape()[i] for i in range(len(tile))]
        return tile_map

    def compute_workload_per_item(self, output_tile) -> float:
        output_node = self.output_nodes[0]
        queue = [(output_node, output_tile)]
        compute = 0
        while len(queue) > 0:
            node, tile = queue.pop(0)
            dep = node.infer_dependency(tile)
            for i, edge in enumerate(node.inputs):
                if not edge.src_node.is_placeholder():
                    subtensor_shape = dep[i]
                    compute += np.prod(subtensor_shape)
                    queue.append((edge.src_node, subtensor_shape))
        return float(compute / np.prod(output_tile))

    def _check_basic_tile(self, output_tile):
        op_tile_map = self.get_tile_map(output_tile)
        out_shape = self.output_nodes[0].get_shape()
        queue = list(op_tile_map.items())
        while len(queue) > 0:
            node, tile = queue.pop(0)
            dep = node.infer_dependency(tile)
            for i, edge in enumerate(node.inputs):
                if not edge.src_node.is_placeholder():
                    subtensor_shape = dep[i]
                    shape = edge.src_node.get_shape()
                    if np.prod(subtensor_shape) / np.prod(shape) != np.prod(output_tile) / np.prod(out_shape):
                        # print(subtensor_shape, shape, output_tile, out_shape)
                        return True
                    if edge.src_node in op_tile_map:
                        assert op_tile_map[edge.src_node] == subtensor_shape
                    else:
                        op_tile_map[edge.src_node] = subtensor_shape
                        queue.append((edge.src_node, subtensor_shape))
        return False

    def get_node_reduce_step_candidates(self, node):
        return {k : get_all_factors(node.raxis[k]) for k in node.raxis}

    def _assign_reduce_step(self, node):
        if len(node.raxis) == 0:
            return {}
        raxis = node.raxis
        tile = node.get_shape()
        all_steps = self.get_node_reduce_step_candidates(node)

        def _score(rstep_id):
            rstep = {k : all_steps[k][rstep_id[k]] for k in rstep_id}
            score = 0
            shape = node.infer_dependency(tile, rstep=rstep)
            for edge in node.inputs:
                if edge.src_node.is_placeholder():
                    read_transaction_elements = 128 // (edge.src_node.get_dtype().bits // 8)
                    score += min(coalesced_factor(shape[edge.dst_id], edge.src_node.get_shape()), read_transaction_elements)
            return score

        def _enlarge(rstep_id):
            candidates = []
            candidates.append((rstep_id, _score(rstep_id)))
            for ax in rstep_id:
                if rstep_id[ax] + 1 == len(all_steps[ax]):
                    continue
                r = rstep_id.copy()
                r[ax] += 1
                candidates.append((r, _score(r)))
            best = max(candidates, key=lambda x:x[1])
            return best

        # enlarge rstep to ensure read is coaleased
        cur_rstep_id = {ax : 0 for ax in raxis}
        cur_score = _score(cur_rstep_id)
        while True:
            if cur_score == 0:break
            new_rstep, new_score = _enlarge(cur_rstep_id)
            if new_score <= cur_score:
                break
            else:
                cur_rstep_id, cur_score = new_rstep, new_score
        rstep = {k : all_steps[k][cur_rstep_id[k]] for k in cur_rstep_id}
        return rstep

    def _expand_reduce_axis(self, output_tile, info, rstep_map):
        tile_map = self.get_tile_map(output_tile)
        _, tile_map = self._compute_memory_footprint(tile_map)
        cur_block_per_SM = info.block_per_SM
        smem_limit = min(self.arch.max_smem_usage // cur_block_per_SM, self.arch.mem_cap(0))
        def _optimize(node, rstep):
            all_steps = self.get_node_reduce_step_candidates(node)

            def _score(rstep_id):
                rstep = {k : all_steps[k][rstep_id[k]] for k in node.raxis}
                score = 0
                shape = node.infer_dependency(tile_map[node], rstep=rstep)
                for edge in node.inputs:
                    if edge.src_node.is_placeholder():
                        factor = coalesced_factor(shape[edge.dst_id], edge.src_node.get_shape())
                        score += factor
                return score

            def _enlarge(rstep_id):
                candidates = []
                for ax in rstep_id:
                    if rstep_id[ax] + 1 == len(all_steps[ax]):
                        continue
                    r = rstep_id.copy()
                    r[ax] += 1
                    candidates.append((r, _score(r)))
                if len(candidates) == 0:
                    return (None, None)
                return max(candidates, key=lambda x:x[1])

            cur_rstep_id = {k : all_steps[k].index(rstep[k]) for k in node.raxis}
            cur_score = _score(cur_rstep_id)
            new_rstep_map = rstep_map.copy()
            while True:
                new_rstep_id, new_score = _enlarge(cur_rstep_id)
                if new_rstep_id is None:
                    break
                new_rstep_map[node] = {k : all_steps[k][new_rstep_id[k]] for k in node.raxis}
                if self._compute_shared_memory_usage(tile_map, new_rstep_map) > smem_limit:
                    break
                else:
                    cur_rstep_id, cur_score = new_rstep_id, new_score
            rstep = {k : all_steps[k][cur_rstep_id[k]] for k in node.raxis}
            return rstep

        rstep_map = rstep_map.copy()
        for node in self.ordered_nodes:
            if len(node.raxis) > 0:
                rstep = _optimize(node, rstep_map[node])
                rstep_map[node] = rstep
        return rstep_map

    def _compute_memory_footprint(self, tile_map):
        tile_map = tile_map.copy()
        queue = [node for node in self.output_nodes]
        footprint = 0
        while len(queue) > 0:
            node = queue.pop(0)
            dep = node.infer_dependency(tile_map[node])
            for i, edge in enumerate(node.inputs):
                if edge.src_node.is_placeholder():
                    read_transaction_elements = 128 // (edge.src_node.get_dtype().bits // 8)
                    footprint += coalesced_tensor_shape(dep[i], edge.src_node.get_shape(), read_transaction_elements)
                elif edge.src_node not in tile_map:
                    tile_map[edge.src_node] = dep[i]
                    queue.append(edge.src_node)
        for node in self.output_nodes:
            write_transaction_elements = 32 // (edge.src_node.get_dtype().bits // 8)
            footprint += coalesced_tensor_shape(tile_map[node], node.get_shape(), write_transaction_elements)
        return footprint, tile_map

    def _compute_shared_memory_usage(self, tile_map, rstep_map):
        allocator = BestFit()
        block_map = {}
        processed = set()
        def can_free(node, out_id):
            for edge in node.outputs:
                if edge.src_id == out_id and edge.dst_node not in processed:
                    return False
            return True
        for node in self.ordered_nodes:
            node_internal_bytes = node.infer_smem_usage(tile_map[node], rstep_map[node])
            block = allocator.malloc(node_internal_bytes)
            allocator.free(block)
            # free inputs
            processed.add(node)
            for edge in node.inputs:
                if not edge.src_node.is_placeholder() and can_free(edge.src_node, edge.src_id):
                    allocator.free(block_map.pop((edge.src_node, edge.src_id)))
            # alloc outputs
            for edge in node.outputs:
                if not edge.dst_node.is_output() and (node, edge.src_id) not in block_map:
                    dtype_bytes = node.get_dtype(edge.src_id).bits // 8
                    block_map[(node, edge.src_id)] = allocator.malloc(np.prod(tile_map[node]) * dtype_bytes)

        assert len(block_map) == 0
        return allocator.limit

    def compute_smem_tile_meta_data(self, output_tile_map, rstep_map) -> TileInfo:
        footprint, tile_map = self._compute_memory_footprint(output_tile_map)
        smem_cost = self._compute_shared_memory_usage(tile_map, rstep_map)
        if smem_cost > self.arch.mem_cap(0):
            return TileInfo(-1, -1, -1, -1)
        out_node = self.output_nodes[0]
        out_tile = output_tile_map[out_node]
        out_shape = out_node.get_shape()
        grid_size = int(np.prod([np.ceil(y / x) for x, y in zip(out_tile, out_shape)]))
        reg_usage = int(2 * max([np.prod(tile_map[node]) for node in self.ordered_nodes])) # estimated reg usage
        if reg_usage > self.arch.reg_cap[0]:
            return TileInfo(-1, -1, -1, -1)
        block_per_SM = min(self.arch.max_smem_usage // max(smem_cost, 1), 4)
        num_wave = int(np.ceil(grid_size / (block_per_SM * self.arch.compute_max_core[0]))) # self.arch.compute_max_core[0]
        return TileInfo(footprint, smem_cost, block_per_SM, num_wave)

    def check_tile_shape_isvalid(self, out_tile):
        output_tile_map = self.get_tile_map(out_tile)
        _, tile_map = self._compute_memory_footprint(output_tile_map)
        out_node = self.output_nodes[0]
        grid_size = np.prod([np.ceil(y / x) for x, y in zip(out_tile, out_node.get_shape())])
        for node in self.ordered_nodes:
            node_grid_size = np.prod([np.ceil(y / x) for x, y in zip(tile_map[node], node.get_shape())])
            if node_grid_size != grid_size:
                return False
        return True

    def recommend_block_size(self, output_tile, rstep_map) -> List[int]:
        tile_map = self.get_tile_map(output_tile)
        _, tile_map = self._compute_memory_footprint(tile_map)
        node_space_sizes = [int(np.prod(tile_map[node])) for node in self.ordered_nodes]
        max_block_size = functools.reduce(math.gcd, node_space_sizes)

        if max_block_size < 128 and max_block_size == min(node_space_sizes):
            node_reduce_sizes = [int(np.prod(list(rstep_map[node].values()))) for node in self.ordered_nodes]
            total_sizes = [x * y for x, y in zip(node_space_sizes, node_reduce_sizes)]
            max_possible_size = functools.reduce(math.gcd, total_sizes)
            possible_block_sizes = list(filter(
                lambda x: x % max_block_size == 0 and x <= 1024, get_all_factors(max_possible_size)))
            possible_block_sizes = list(filter( # either be a factor of space or cover fully cover the space
                lambda x: all([x % s == 0 or s % x == 0 for s in node_space_sizes]) , possible_block_sizes))
            factor_ordered = sorted(possible_block_sizes, key=score_block_size)
            return factor_ordered
        else:
            possible_block_sizes = get_all_factors(max_block_size)
            possible_block_sizes = list(filter(lambda x: x <= 1024, possible_block_sizes))
        factor_ordered = sorted(possible_block_sizes, key=score_block_size)
        return factor_ordered

    def assign_block_size(self, output_tile, rstep_map, topk=1) -> Generator[Dict, Node, Config]:
        tile_map = self.get_tile_map(output_tile)
        _, tile_map = self._compute_memory_footprint(tile_map)
        block_size_ordered = self.recommend_block_size(output_tile, rstep_map)
        for block_size in block_size_ordered:
            result = {}
            failed = False
            for node in self.ordered_nodes:
                result[node] = self._assign_block_size(node, tile_map[node], rstep_map[node], block_size)
                if result[node] is None:
                    failed = True
                    break
            if failed:
                continue
            else:
                yield result
                topk -= 1
                if topk == 0:
                    break

    def _assign_block_order(self, output_tile):
        tile_map = self.get_tile_map(output_tile)
        _, tile_map = self._compute_memory_footprint(tile_map)
        queue = [node for node in self.output_nodes]
        block_idx = tvm.te.var("block_idx")
        block_idx_map = {node : block_idx for node in self.output_nodes}
        result = {}
        while len(queue) > 0:
            node = queue.pop(0)
            if node.is_output():
                block_idx_map[node.inputs[0].src_node] = block_idx_map[node]
                queue.append(node.inputs[0].src_node)
                continue

            deps = node.block_infer(tile_map, block_idx_map[node], block_idx)
            for i, edge in enumerate(node.inputs):
                if edge.src_node.is_placeholder():
                    continue
                elif edge.src_node not in block_idx_map:
                    block_idx_map[edge.src_node] = deps[i]
                    queue.append(edge.src_node)
                    if not (deps[i].same_as(block_idx) or isinstance(deps[i], tvm.tir.expr.ConstExpr)):
                        result[edge.src_node] = deps[i]
        return result

    def _assign_block_size(self, node: Node, tile, rstep_map, block_size):
        factors = factorize(block_size)
        cur_threads = [1 for _ in tile]
        reduce_thread = {k : 1 for k in rstep_map}
        ndim = len(tile)

        def _score(node, thread): # small is better
            score = 0
            block_tile = [int(np.ceil(tile[i] / thread[i])) for i in range(ndim)]
            shape = node.infer_dependency(block_tile)
            for edge in node.inputs:
                score += np.prod(shape[edge.dst_id]) / self.arch.memory_bw(1)
            for edge in node.outputs:
                if edge.dst_node in self.output_nodes: # write to global
                    score += coalesced_tensor_shape(thread, node.get_shape(), 8) / self.arch.memory_bw(0)
            return score
        for factor in reversed(factors):
            score_map = {}
            for i in range(ndim):
                if cur_threads[i] >= tile[i]:
                    continue
                if (tile[i] % (cur_threads[i] * factor)) != 0:
                    continue
                cur_threads[i] *= factor
                score_map[i] = (_score(node, cur_threads), i)
                cur_threads[i] //= factor
            if len(score_map) > 0:
                # assign to space axis
                dim_order = sorted(score_map.keys(), key=lambda x:score_map[x])
                cur_threads[dim_order[0]] *= factor
            else:
                # assign to reduce axis
                target_ax = None
                for ax, ax_len in reversed(list(rstep_map.items())):
                    if ax_len % (reduce_thread[ax] * factor) == 0:
                        target_ax = ax
                        break
                assert target_ax
                reduce_thread[target_ax] *= factor

        codegen_dict = Config()
        codegen_dict.block = tile
        codegen_dict.thread = cur_threads
        codegen_dict.rstep = [rstep_map[ax] for ax in node.raxis]
        codegen_dict.reduce_thread = [reduce_thread[ax] for ax in node.raxis]
        # if node.get_dtype().bits == 16:
        #     codegen_dict._step = [1 for _ in range(ndim)]
        #     for i in range(ndim):
        #         if codegen_dict.block[i] // codegen_dict.thread[i] % 2 == 0:
        #             codegen_dict._step[i] = 2
        # if len(rstep_map) > 0 and np.prod(block_tile) * np.prod(list(rstep_map.values())) < 1000:
        #     codegen_dict["unroll"] = True
        # # assign virtual threads
        # codegen_dict = {}
        # out_shape = node.get_shape()
        # for i, ax in enumerate(node.saxis):
        #     strided = coalesced_tensor_shape(cur_threads[i:], out_shape[i:], 8)
        #     unstrided = cur_threads[i]
        #     if i + 1 < len(node.saxis):
        #         unstrided *= coalesced_tensor_shape(cur_threads[i+1:], out_shape[i:], 8)
        #     else:
        #         unstrided *= 8
        #     if strided < unstrided:
        #         codegen_dict[ax] = [block_tile[i], cur_threads[i], 1]
        #     else:
        #         codegen_dict[ax] = [1, cur_threads[i], block_tile[i]]

        # # assign reduce order
        # # more local memory reuse between two steps is ordered as inner loop
        # if len(node.raxis) > 0:
        #     thd_tile = [codegen_dict[ax][-1] for ax in node.saxis]
        #     ax_score = {}
        #     for i, rax in enumerate(node.raxis):
        #         rstep = {ax : 1 for ax in node.raxis}
        #         rstep[rax] = min(2, node.raxis[rax])
        #         ax_score[rax] = node.infer_reduction_inputs(thd_tile, rstep)
        #     axis_order = sorted(ax_score.keys(), key=lambda ax: ax_score[ax], reverse=True)
        #     codegen_dict["raxis_order"] = axis_order
        return codegen_dict
