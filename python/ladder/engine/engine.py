# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import List
import logging

from ..graph import Node, find_topo_sort_priority
from ..utils import CompileResult
from .base_tunner import Tunner
from .common import FusionGroup
from tqdm import tqdm

logger = logging.getLogger(__name__)

def _get_nodes_dependency(nodes: List[Node], processed: List[Node]) -> List[Node]:
    """
        returns dependency for input nodes (not in processed, not placeholder),
        will include input nodes themself.
    Args:
        nodes: target nodes to infer dependency
        processed : already done nodes
    """
    queue = list(nodes)
    deps = set()
    while len(queue) > 0:
        node = queue.pop(0)
        deps.add(node)
        for edge in node.inputs:
            if edge.src_node.is_placeholder():
                continue
            if edge.src_node in processed or edge.src_node in deps:
                continue
            queue.append(edge.src_node)
    return list(deps)

class Engine:
    def __init__(self, tunner: Tunner) -> None:
        self.tunner = tunner

    def set_debug_nodes(self, ordered_nodes, names):
        nodes = []
        for node in ordered_nodes:
            if node.name in names:
                nodes.append(node)
        return nodes

    def run(self, ordered_nodes: List[Node]) -> List[FusionGroup]:
        output_list = list(filter(lambda node : node.is_output(), ordered_nodes))
        ordered_nodes = find_topo_sort_priority(output_list)
        '''
            for debug purpose, only run some nodes can be used through:
            ordered_nodes = self.set_debug_nodes(ordered_nodes, ['ladder_perfect_matmul_29', 'layout_transform_reshape_reshape_add_30'])
        '''
        logger.debug("tune candidate nodes: %s", ordered_nodes)
        
        self.node2group = {} # map node to fused group
        self.node_topo_id = {ordered_nodes[i] : i for i in range(len(ordered_nodes))}
        fusion_groups = []
        for node in tqdm(ordered_nodes, desc="Processing", ascii=False,
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"):
            if node in self.node2group or node.is_output() or node.is_placeholder():
                continue
            fg = self._build_fusion_group(node)
            fusion_groups.append(fg)
            logger.info(f"Fusion group created: {fg.group_id} {[node.name for node in fg.nodes]}")
        return fusion_groups

    def run_no_fusion(self, ordered_nodes: List[Node]) -> List[FusionGroup]:
        fusion_groups = []
        group_id = 0
        for node in ordered_nodes:
            if node.is_output() or node.is_placeholder():
                continue
            result = self.tunner.tune([node], kernel_name=node.name)
            fusion_groups.append(FusionGroup([node], group_id, result, 0))
            group_id += 1
        return fusion_groups

    def _build_fusion_group(self, top_node):
        cur_group = [top_node]
        cur_group_id = 0 if len(self.node2group) == 0 else max(self.node2group.values()) + 1
        cur_latency_gain = 0
        self.node2group[top_node] = cur_group_id
        queue = [(top_node, i) for i in range(top_node.num_outputs())]
        cp_result = None
        while len(queue) > 0:
            node, output_id = queue.pop(0)
            fusing_nodes = []
            valid = True
            for edge in node.outputs:
                if edge.src_id != output_id:
                    continue
                if edge.dst_node.is_output(): # model output can't be eliminated
                    valid = False
                    break
                if edge.dst_node in fusing_nodes or edge.dst_node in cur_group:
                    continue
                assert edge.dst_node not in self.node2group
                fusing_nodes.append(edge.dst_node)

            if not valid:
                continue

            fusing_nodes = _get_nodes_dependency(fusing_nodes, self.node2group)
            if len(fusing_nodes) == 0 or len(fusing_nodes) > 10: # too many dependency
                continue

            new_group = fusing_nodes + cur_group # create a new subgraph candidate

            # checking group output is valid
            in_group_outputs, out_group_outputs = set(), set()
            for node in new_group:
                for edge in node.outputs:
                    if edge.dst_node in new_group:
                        in_group_outputs.add((node, edge.src_id))
                    else:
                        out_group_outputs.add((node, edge.src_id))
            if in_group_outputs.intersection(out_group_outputs):
                continue

            new_group = sorted(new_group, key=lambda n:self.node_topo_id[n])
            result = self.tunner.tune(new_group, kernel_name="Group"+str(cur_group_id))
            # try experimental local fuse
            if len(fusing_nodes) == 1 and len(cur_group) == 1:
                if not fusing_nodes[0].get_tag("skip") and not cur_group[0].get_tag("skip") and \
                    fusing_nodes[0].reduce_op is None and cur_group[0].reduce_op is not None:
                    connection = [[cur_group[0].name, fusing_nodes[0].name]]
                    result_local = self.tunner.tune(new_group, connection, kernel_name="Group"+str(cur_group_id))
                    if result_local and (result is None or result_local.latency < result.latency):
                        result = result_local
            if result is None:
                continue
            gain = self.compute_gain(new_group, result)
            if gain < cur_latency_gain:
                continue
            cur_latency_gain = gain
            cur_group = new_group
            cp_result = result
            for n in fusing_nodes:
                self.node2group[n] = cur_group_id
                for i in range(n.num_outputs()):
                    queue.append((n, i))

        if cp_result is None: # tune single op if no fusion is possible
            assert len(cur_group) == 1
            if not top_node.get_tag("skip"):
                cp_result = self.tunner.tune(cur_group, kernel_name="Group"+str(cur_group_id))
                if cp_result is None:
                    logger.error("Cannot generate code for " + top_node.name)
            else:
                logger.info("Skipping node " + top_node.name)
        return FusionGroup(cur_group, cur_group_id, cp_result, cur_latency_gain)

    def compute_gain(self, group: List[Node], cp_result: CompileResult) -> float:
        for node in group:
            if node.get_tag("latency") is None:
                if node.get_tag("memcpy"):
                    node.add_tag("latency", 0)
                    continue
                result = self.tunner.tune([node], kernel_name=node.name)
                if result is None:
                    latency = 1e8
                else:
                    latency = result.latency
                node.add_tag("latency", latency)
        base = sum([node.get_tag("latency") for node in group])
        new = cp_result.latency
        return base - new
