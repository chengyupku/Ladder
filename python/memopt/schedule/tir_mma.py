import numpy as np
from tvm import tir

from ..config import Stride
from ..IRpass import ApplyLayoutPass
from ..layout import *
from .cutlass_intrin import *
from .tir_base import TIRSchedulerBase


class TIRCutlassMMAScheduler(TIRSchedulerBase):
    def schedule(self) -> tir.Schedule:
        sch, config = self.sche, self.config
        self.block_size[0] = 32
        self.block_size[1] = int(np.prod(self.config.block)) // int(np.prod(self.config.warp))
        C = sch.get_block(self.reduce_op.name)
        space_loops = sch.get_loops(C)[:len(self.reduce_op.axis)]
        assert(len(self.reduce_op.reduce_axis) == 1)
        ax_K = sch.get_loops(C)[-1]
        A_ax_m, A_ax_k, B_ax_k, B_ax_n, C_ax_m, C_ax_n = config.tc_extra_conf.tc_axis
        transpose_A = A_ax_k < A_ax_m
        transpose_B = B_ax_k > B_ax_n
        block_tile_M, block_tile_N = self.config.block[C_ax_m], self.config.block[C_ax_n]
        warp_tile_M, warp_tile_N = self.config.warp[C_ax_m], self.config.warp[C_ax_n]
        out_dtype = self.reduce_op.output(0).dtype
        in_dtype = self.reduce_op.input_tensors[0].dtype

        # ------------------------ Block and Warp level job partition ------------------------

        block_axis = []
        warp_axis = []
        inner_axis = []
        for i, loop in enumerate(space_loops):
            if i in (C_ax_m, C_ax_n):
                bo, wo, wi = sch.split(loop, factors=[None, config.block[i] // config.warp[i], config.warp[i]])
                block_axis.append(bo)
                warp_axis.append(wo)
                inner_axis.append(wi)
            else:
                assert config.block[i] == 1
                block_axis.append(loop)

        chunk_size = config.rstep[0]
        K_outer, K_inner = sch.split(ax_K, factors=[None, chunk_size])

        sch.reorder(*block_axis, *warp_axis, K_outer, *inner_axis, K_inner)
        block_fused = sch.fuse(*block_axis)
        warp_fused = sch.fuse(*warp_axis)
        sch.bind(block_fused, "blockIdx.x")
        sch.bind(warp_fused, "threadIdx.y")

        # ------------------------ Shared memory layout for multiplicand A and B ------------------------
        try:
            if transpose_A:
                layoutA = ColumnMajorVoltaTensorOpMultiplicandCongruous(chunk_size, block_tile_M)
            else:
                layoutA = RowMajorVoltaTensorOpMultiplicandCrosswise(block_tile_M, chunk_size)
            if transpose_B:
                layoutB = ColumnMajorVoltaTensorOpMultiplicandCrosswise(block_tile_N, chunk_size)
            else:
                layoutB = RowMajorVoltaTensorOpMultiplicandBCongruous(chunk_size, block_tile_N)
        except AssertionError:
            if transpose_A:
                layoutA = ColumnMajorLayout(chunk_size, block_tile_M)
            else:
                layoutA = RowMajorLayout(block_tile_M, chunk_size)
            if transpose_B:
                layoutB = ColumnMajorLayout(block_tile_N, chunk_size)
            else:
                layoutB = RowMajorLayout(chunk_size, block_tile_N)

        AS = sch.cache_read(C, 0, "shared")
        BS = sch.cache_read(C, 1, "shared")
        sch.compute_at(AS, K_outer)
        sch.compute_at(BS, K_outer)

        A_stride, B_stride = Stride(), Stride()
        if layoutA.requires_padding():
            A_high_ax = min(A_ax_m, A_ax_k)
            padA = 8 if transpose_A else 4
            layoutA.set_pad(padA)
            A_stride = Stride(int(np.prod(config.tc_extra_conf.AS_shape[A_high_ax+1:])) + padA, A_high_ax)
        if layoutB.requires_padding():
            B_high_ax = min(B_ax_n, B_ax_k)
            padB = 4 if transpose_B else 8
            layoutB.set_pad(padB)
            B_stride = Stride(int(np.prod(config.tc_extra_conf.BS_shape[B_high_ax+1:])) + padB, B_high_ax)
        # dim_offset = 3 (block_fused, warp_fused, K_outer)
        self.cooperative_fetch(AS, 3, A_stride, inner_step=layoutA.get_vectorize())
        self.cooperative_fetch(BS, 3, B_stride, inner_step=layoutB.get_vectorize())

        # ------------------------ Schedule output fragment layout ------------------------

        C_warp = sch.cache_write(C, 0, "cutlass.warp.mma")
        sch.reverse_compute_at(C_warp, warp_fused)
        block_init_c = sch.decompose_reduction(C, sch.get_loops(C)[2])
        layoutC = voltaFragmentCLayout32x32(warp_tile_M, warp_tile_N)
        sch.transform_loop(C_warp, 2, layoutC)
        sch.bind(sch.get_loops(C_warp)[-2], "threadIdx.x")
        oo, vec = sch.split(sch.get_loops(C_warp)[-1], factors=[None, layoutC.get_vectorize()])
        sch.vectorize(vec)
        sch.unroll(oo)
        self.schedule_compute_inline()

        # ------------------------ Tensorize and Pipelining -------------------------

        cls_code = register_volta_cutlass_warp_mma(warp_tile_M, warp_tile_N, chunk_size, layoutA, layoutB)
        sch.tensorize(sch.get_loops(block_init_c)[-2],
            register_cutlass_warp_init_intrin(warp_tile_M, warp_tile_N, out_dtype,
            cls_code, block_tile_M // warp_tile_M, block_tile_N // warp_tile_N)
        )
        sch.tensorize(sch.get_loops(C)[-3],
            register_gemm_intrin(
                config.warp[C_ax_m], config.warp[C_ax_n], chunk_size,
                in_dtype, out_dtype,
                transpose_A, transpose_B,
                layoutA, layoutB)
        )

        sch.annotate(AS, "tir.manifest_shared_memory_local_stage", 1)
        sch.annotate(BS, "tir.manifest_shared_memory_local_stage", 1)
        sch.annotate(K_outer, "software_pipeline_stage", [0, 0, 0, 0, 1])
        sch.annotate(K_outer, "software_pipeline_order", [0, 3, 1, 4, 2])

        layout_pass = ApplyLayoutPass({
            self.reduce_op.input_tensors[0].name+"_shared": layoutA,
            self.reduce_op.input_tensors[1].name+"_shared": layoutB,
            self.reduce_op.name + "_cutlass.warp.mma": layoutC.fragment_offset})
        self.passes.append(layout_pass.get_pass())

        # self.debug_schedule()
        # exit(0)

        return sch.mod["main"]

    def build(self, target) -> str:
        with tvm.transform.PassContext(config={"tir.add_lower_pass": self.passes}, disabled_pass=["tir.UnrollLoop"]):
            mod = tvm.build(self.sche.mod["main"], self.args, target)
        return mod.imported_modules[0].get_source()