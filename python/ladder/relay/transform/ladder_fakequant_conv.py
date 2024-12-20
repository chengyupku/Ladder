# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import tvm
from tvm import relay, ir
import numpy as np


"""
   Fake Quant Conv.
"""


@relay.transform.function_pass(opt_level=0, required=["InferType"])
class LadderFakeQuantConv(relay.ExprMutator):
    def __init__(self, quant_weight_candidate=None, quant_config=None, quant_type=0, convert_int=False, convert_float=False):
        super().__init__()
        """
        quant_gemm_candidate: list of weight candidates
            (
                (N, K, is_transpose),
                (N, K, is_transpose),
                ...
            )
            if None, quantize all the weight 

        quant_type:
            0: qweight
            1: qweight + scales
            2: qweight + scales + zeros
        quant_config:
            {
                'format':'nf',
                'bits': args.bits,
                'group_size': -1,
            }
        """
        self.quant_weight_candidate = quant_weight_candidate
        self.quant_type = quant_type
        self.quant_config = quant_config
        self.convert_int = convert_int
        self.convert_float = convert_float

    def transform_function(self, func, mod, ctx):
        return self.visit(func)

    def visit_call(self, call):
        if isinstance(call.op, ir.Op) and call.op.name in [
            "ladder.perfect_im2col_conv",
        ]:
            for type in call.type_args:
                if type.dtype != "float16":
                    return super().visit_call(call)

            data = self.visit(call.args[0])
            kernel = self.visit(call.args[1])
            input_shape = call.args[0].checked_type.shape
            kernel_shape = call.args[1].checked_type.shape
            n, h, w, c, nn, cc = input_shape
            k, j, kk, jj = kernel_shape
            out_dtype = call.checked_type.dtype
            
            other_inputs = []
            conv2d_attrs = call.attrs
            
            if self.convert_float:
                quant_data = relay.cast(data, "float32")
                quant_kernel = relay.cast(kernel, "float32")

                attrs = ir.make_node(
                    "DictAttrs",
                    out_dtype='float32',
                    strides=conv2d_attrs.strides,
                    padding=conv2d_attrs.padding,
                    dilation=conv2d_attrs.dilation,
                    data_layout=conv2d_attrs.data_layout,
                    kernel_layout=conv2d_attrs.kernel_layout,
                    kernel_size=conv2d_attrs.kernel_size,
                    can_propagate=conv2d_attrs.can_propagate,
                )
                q_perfect_conv = relay.Call(
                    relay.op.get("ladder.perfect_im2col_conv"),
                    [quant_data, quant_kernel],
                    attrs,
                )
                q_perfect_conv = relay.cast(q_perfect_conv, out_dtype)
                return q_perfect_conv

            # convert kernel to quant format shape -> (N, K // 2), dtype -> int8
            # assume the kernel is stored with transpose. (int8 mma tensorcore only support nt case.)
            if self.quant_config['format'] == 'mxfp':
                block_scale_data = tvm.nd.array(
                    np.random.randint(0, 127, (int(k) * int(kk) // 32, int(jj) * int(j))).astype(np.uint8)
                )
                block_scale = relay.const(block_scale_data)
                other_inputs.append(block_scale)
                quant_kernel_shape = (
                    (int(j), int(k), int(jj), int(kk) // 8 * self.quant_config["bits"])
                )
                quant_kernel_data = tvm.nd.array(
                    np.random.randint(
                        low=np.iinfo(np.int8).min,
                        high=np.iinfo(np.int8).max + 1,
                        size=quant_kernel_shape,
                        dtype=np.int8,
                    )
                )
                quant_kernel = relay.const(quant_kernel_data)

                attrs = ir.make_node(
                    "DictAttrs",
                    out_dtype='float32',
                    strides=conv2d_attrs.strides,
                    padding=conv2d_attrs.padding,
                    dilation=conv2d_attrs.dilation,
                    data_layout=conv2d_attrs.data_layout,
                    kernel_layout=conv2d_attrs.kernel_layout,
                    kernel_size=conv2d_attrs.kernel_size,
                    **self.quant_config
                )
                q_perfect_conv = relay.Call(
                    relay.op.get("ladder.perfect_im2col_quant_conv"),
                    [data, quant_kernel, *other_inputs],
                    attrs,
                )
                q_perfect_conv = relay.cast(q_perfect_conv, out_dtype)
                return q_perfect_conv

            if self.quant_config['format'] == 'int4b':
                # special case for int4b tensor core
                # input should be cast into int8
                quant_data = relay.cast(data, "float32")
                quant_data = relay.cast(quant_data, "int8")
                # should set wmma_k -> 32
                quant_data = relay.reshape(quant_data, (n, h, w, c // 2, nn, 32))
                # relay implemetation for crop
                quant_data = relay.strided_slice(quant_data, begin=[0, 0, 0, 0, 0, 0], end=[n, h, w, c // 4, nn, 32])

                quant_kernel_shape = (
                    (int(j), int(k) // 4, int(jj), int(kk) // 2 * self.quant_config["bits"])
                )
                quant_kernel_data = tvm.nd.array(
                    np.random.randint(
                        low=np.iinfo(np.int8).min,
                        high=np.iinfo(np.int8).max + 1,
                        size=quant_kernel_shape,
                        dtype=np.int8,
                    )
                )

                quant_kernel = relay.const(quant_kernel_data)

                other_inputs = [relay.cast(input, "int8") for input in other_inputs]
                attrs = ir.make_node(
                    "DictAttrs",
                    out_dtype='int32',
                    strides=conv2d_attrs.strides,
                    padding=conv2d_attrs.padding,
                    dilation=conv2d_attrs.dilation,
                    data_layout=conv2d_attrs.data_layout,
                    kernel_layout=conv2d_attrs.kernel_layout,
                    kernel_size=conv2d_attrs.kernel_size,
                    **self.quant_config
                )
                q_perfect_conv = relay.Call(
                    relay.op.get("ladder.perfect_im2col_quant_conv"),
                    [quant_data, quant_kernel, *other_inputs],
                    attrs,
                )
                q_perfect_conv = relay.cast(q_perfect_conv, out_dtype)
                return q_perfect_conv
           
            if self.convert_int:
                quant_data = relay.cast(data, "float32")
                quant_data = relay.cast(quant_data, "int8")
                quant_data = relay.reshape(quant_data, (n, h, w, -1, nn, 32))
                quant_kernel_shape = (
                    (int(j), int(k) // 2, int(jj), int(kk) // 4 * self.quant_config["bits"])
                )
                quant_kernel_data = tvm.nd.array(
                    np.random.randint(
                        low=np.iinfo(np.int8).min,
                        high=np.iinfo(np.int8).max + 1,
                        size=quant_kernel_shape,
                        dtype=np.int8,
                    )
                )

                quant_kernel = relay.const(quant_kernel_data)

                other_inputs = [relay.cast(input, "int8") for input in other_inputs]
                attrs = ir.make_node(
                    "DictAttrs",
                    out_dtype='int32',
                    strides=conv2d_attrs.strides,
                    padding=conv2d_attrs.padding,
                    dilation=conv2d_attrs.dilation,
                    data_layout=conv2d_attrs.data_layout,
                    kernel_layout=conv2d_attrs.kernel_layout,
                    kernel_size=conv2d_attrs.kernel_size,
                    **self.quant_config
                )
                q_perfect_conv = relay.Call(
                    relay.op.get("ladder.perfect_im2col_quant_conv"),
                    [quant_data, quant_kernel, *other_inputs],
                    attrs,
                )
                q_perfect_conv = relay.cast(q_perfect_conv, out_dtype)
            else:   
                quant_kernel_shape = (
                    (int(j), int(k), int(jj), int(kk) // 8 * self.quant_config["bits"])
                )
                quant_kernel_data = tvm.nd.array(
                    np.random.randint(
                        low=np.iinfo(np.int8).min,
                        high=np.iinfo(np.int8).max + 1,
                        size=quant_kernel_shape,
                        dtype=np.int8,
                    )
                )

                quant_kernel = relay.const(quant_kernel_data)
                attrs = ir.make_node(
                    "DictAttrs",
                    out_dtype=out_dtype,
                    strides=conv2d_attrs.strides,
                    padding=conv2d_attrs.padding,
                    dilation=conv2d_attrs.dilation,
                    data_layout=conv2d_attrs.data_layout,
                    kernel_layout=conv2d_attrs.kernel_layout,
                    kernel_size=conv2d_attrs.kernel_size,
                    **self.quant_config
                )
                q_perfect_conv = relay.Call(
                    relay.op.get("ladder.perfect_im2col_quant_conv"),
                    [data, quant_kernel, *other_inputs],
                    attrs,
                )
            return q_perfect_conv
    
        return super().visit_call(call)
