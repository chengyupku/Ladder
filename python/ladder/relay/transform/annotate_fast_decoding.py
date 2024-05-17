# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from tvm import relay

from .utils import check_tensor_core_valid_shape

@relay.transform.function_pass(opt_level=0)
class AnnotateFastDecoding(relay.ExprMutator):
    def __init__(self):
        super().__init__()

    def transform_function(self, func, mod, ctx):
        return super().visit_function(func)

    def visit_function(self, func):
        visitor = OpVisitor()
        visitor.visit(func)
        if visitor.fast_decoding is not None:
            func = func.with_attr("fast_decoding", visitor.fast_decoding)
        if visitor.consistent_config is not None:
            func = func.with_attr("consistent_config", visitor.consistent_config)
        return super().visit_function(func)

class OpVisitor(relay.ExprVisitor):
    def __init__(self):
        super().__init__()
        self.fast_decoding = None
        self.consistent_config = None

    def visit_call(self, call):
        if call.op.name in ["ladder.quant_linear", "ladder.perfect_quant_linear", "perfect_im2col_quant_conv"]:
            self.fast_decoding = True
            self.consistent_config = (True, False)
        return super().visit_call(call)

__all__ = ["AnnotateFastDecoding"]
