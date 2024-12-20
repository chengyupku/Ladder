import tvm
import ladder
from ladder.graph import IRNode, OutputNode
from ladder.policy import *
from tvm import relay
import os.path as osp
from tvm.contrib.target.onnx import to_onnx
from tvm.relay.testing import run_infer_type
from tvm.contrib import graph_executor
import os
from tvm.script import tir as T
from tvm import te
from ladder.reference import get_subgraph_reference_outputs
import numpy as np
import logging

ladder.set_log_level(logging.DEBUG)
# get file name and remove the suffix
fname = os.path.basename(__file__)
fname = os.path.splitext(fname)[0]
# create log path
log_path = "progress/" + fname

arch = "cuda"
arch = ladder.arch.__getattribute__(arch)()
dtype="float16"

shapes = [
    # [1, 1024, 8192], 
    # [1, 8192, 8192], 
    # [1, 8192, 28672], 
    # [1, 28672, 8192], 
    [1, 16384, 16384]
]
ft_kernel = [
    [1, 9216, 12288],
    [1, 12288, 12288],
    [1, 12800, 12288],
    [1, 12288, 3072],
]
llm_shapes = [
    [1, 16384, 16384],
    [1, 43008, 14336],
    [1, 14336, 14336],
    [1, 57344, 14336],
    [1, 14336, 57344],
    [1, 9216, 9216],
    [1, 36864, 9216],
    [1, 9216, 36864],
    [1, 22016, 8192],
    [1, 8192, 22016],
    [1, 8192, 8192],
    [1, 28672, 8192],
    [1, 8192, 22016],
]
# shapes = llm_shapes
perf_map = {}
for M, N, K in shapes:

    bit = 2
    n_float_per_i8 = 8 // bit
    mask = (1 << bit) - 1


    def _tir_u8_to_int_to_float(nbit: int, val: tvm.tir.PrimExpr, pos: tvm.tir.PrimExpr, dtype: str):
        assert val.dtype == "int8"
        mask = tvm.tir.const((1 << nbit) - 1, "int8")
        return ((val >> (pos * nbit).astype("int8")) & mask).astype(dtype)
        

    A = te.placeholder((M, K), name='A', dtype='float16')
    B = te.placeholder((N, K // 8 * bit), name='B', dtype='int8')

    def decode_func(n, k):
        w = _tir_u8_to_int_to_float(bit, B[n, k // n_float_per_i8], k % n_float_per_i8, "float16")
        return w

    B_decode = te.compute(
        (N, K),
        decode_func,
        name='B_decode'
    )
    

    # Describe the matrix multiplication in TE
    k = te.reduce_axis((0, K), name='k')
    C = te.compute(
        (M, N),
        lambda i, j: te.sum(A[i, k] * B_decode[j, k], axis=k),
        name='C'
    )

    input_args = [A, B]
    output_args = [C]
    node = IRNode([None for _ in input_args], input_args+output_args, "ladder_matmul")
    node.add_tag("consistent_config", (True, False))
    node.add_tag("fast_decoding", True)
    output_nodes = [OutputNode(node)]
    policy = DefaultPolicy(output_nodes, arch)
    configs = policy.emit_config(20)

    compile_results = []
    cgen = ladder.CodeGenerator()
    for config in configs:
        cpresult = cgen.compile(output_nodes, config, "cuda", kernel_name="Fused")
        compile_results.append(cpresult)
        print(cpresult.grid_size)
        print(cpresult.block_size)
    ladder.utils.compile_and_load_parallel(compile_results, arch)
    best_latency = 10000
    best = None
    values = []
    for cpresult in compile_results:
        print(cpresult.config)
        code = cpresult.code
        if cpresult.lib is None:
            latency = 10000
        else:
            latency = cpresult.profile()
        values.append(latency)
        if latency < best_latency:
            best_latency = latency
            best = cpresult
        print(latency)
        
    print("top1: {} \ttop10: {}".format(values[0], min(values)))
    print("-" * 80, flush=True)
    print("best config: {}".format(best.config))
    print("best latency: {}".format(best_latency))
    # with open("int2.cu", "w") as f:
    #     f.write(best.code)
    print("best code: ", best.code)
        
    key = "{}_{}_{}".format(M, N, K)
    perf_map[key] = best_latency
    if False:
        out = best.get_example_outputs()
        ref_out = get_subgraph_reference_outputs(output_nodes)
        print("out last 10: ", out[:10])
        print("ref_out last 10: ", ref_out[:10])
        for a, b in zip(out, ref_out):
            diff = np.max(np.abs(a-b))
            print("value diff:", diff)

for M, N, K in shapes:
    key = "{}_{}_{}".format(M, N, K)
    print("{}\t{}".format(key, perf_map[key]))
