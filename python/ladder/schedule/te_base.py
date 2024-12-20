# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from typing import List

import numpy as np
import tvm
from tvm import te

from .scheduler_base import SchedulerBase
from ..config import Stride
from ..IRpass import *
from ..te_utils import create_proxy_output


class TESchedulerBase(SchedulerBase):
    def cooperative_fetch(
        self,
        shared,
        strides: Stride = Stride(),
        inner_step: int = 1,
        vectorize_inner=True,
    ):
        assert self.block_size[2] == 1
        axes = self.sche[shared].op.axis
        if strides.is_valid():
            self.sche[shared].storage_align(
                axes[strides.ax], strides.stride - 1, strides.stride
            )
        fused = self.sche[shared].fuse(*axes)
        fused, tv = self.sche[shared].split(fused, factor=inner_step)
        _t, tx = self.sche[shared].split(fused, factor=self.block_size[0])
        oo, ty = self.sche[shared].split(_t, factor=self.block_size[1])
        self.sche[shared].reorder(oo, ty, tx)
        if vectorize_inner:
            self.sche[shared].vectorize(tv)
        # else:
        #     self.sche[shared].unroll(tv)
        # self.sche[shared].unroll(oo)
        self.sche[shared].bind(tx, te.thread_axis("threadIdx.x"))
        self.sche[shared].bind(ty, te.thread_axis("threadIdx.y"))

    def requires_cache(self, tensor, op):
        if tensor in self.shared_inputs:
            return True
        return tensor.name in self.config.cached_tensors

    def create_schedule(self) -> te.Schedule:
        sche = te.create_schedule([self.output_op])
        # use the op reference in te.schedule to avoid bugs
        self.ops = []
        for op in sche.stage_map:
            if isinstance(op, te.ComputeOp):
                if op == self.output_op:
                    self.output_op = op
                if op == self.reduce_op:
                    self.reduce_op = op
                self.ops.append(op)
        return sche

    def build(self, target) -> str:
        with tvm.transform.PassContext(
            config={"tir.add_lower_pass": self.passes, "tir.disable_cse_tir": True}
        ):
            mod = tvm.build(self.sche, self.args, target=target)
        return mod.imported_modules[0].get_source()

    def make_passes(self) -> None:
        self.passes.append(
            RewriteOutputPass(
                self.shared_outputs, self.config.output_strides, self.config.block, True
            ).get_pass()
        )
        self.passes.append(RewriteInputPass(self.shared_inputs, True).get_pass())
        self.passes.append(FixCudaCastPass().get_pass())

    def get_mod_script(self) -> str:
        return str(tvm.lower(self.sche, self.args, simple_mode=True))

    def proxy_outputs(self, output_args) -> List[te.Tensor]:
        if len(output_args) > 1:
            output_args = create_proxy_output(output_args)
        return output_args
