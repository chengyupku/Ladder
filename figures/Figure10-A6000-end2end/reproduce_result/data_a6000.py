# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
llama2_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
llama2_times_data = [
    ("PyTorch-Inductor", [2.687621117, 2.79825449, 79.96388912]),
    ("ONNXRuntime", [2.6808, 2.9429, 104.9512]),
    ("TensorRT", [2.50583, 2.58743, 101.032]),
    ("Welder", [2.54144, 2.689024, 76.433922]),
    ("vLLM", [2.578270435, 2.654864788, 64.82847691]),
    ("vLLM-W$_{INT4}$A$_{FP16}$", [0.79734087, 1.550245285, 196.6294551]),
    ("Bitter", [2.4559, 2.5785, 66.4663]),
    ("Bitter-W$_{INT4}$A$_{FP16}$", [0.657948069, 1.185360816, 64.75624679]),
    ("Bitter-W$_{NF4}$A$_{FP16}$", [0.681704869, 1.338692052, 66.25319969]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [1.256130442, 1.475792808, 64.78081631]),
    ("Bitter-W$_{MXFP8}$A$_{MXFP8}$", [1.342351263, 1.789379982, 82.90916215]),
    ("Bitter-W$_{INT1}$A$_{INT8}$", [0.209039266, 0.752349635, 41.1277536]),
]

bloom_providers = ["BS1 SEQ1", "BS32 SEQ1", "BS1 SEQ4096"]
bloom_times_data = [
    ("PyTorch-Inductor", [7.357809544, 7.336108685, 269.2148328]),
    ("ONNXRuntime", [7.337, 8.6553, 990.0948]),
    ("TensorRT", [7.01336, 7.22799, 244.088]),
    ("Welder", [7.35232, 7.8592, 257.027069]),
    ("vLLM", [7.10080862, 7.397177219, 280.0931311]),
    ("vLLM-W$_{INT4}$A$_{FP16}$", [2.459495068, 4.799568653, 689.5027256]),
    ("Bitter", [6.9831, 7.1805, 204.0919]),
    ("Bitter-W$_{INT4}$A$_{FP16}$", [1.806397709, 2.509997322, 197.1157714]),
    ("Bitter-W$_{NF4}$A$_{FP16}$", [1.838695201, 2.923462986, 201.2320358]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [3.52866644, 3.741222947, 196.9285823]),
    ("Bitter-W$_{MXFP8}$A$_{MXFP8}$", [3.667234483, 4.147930234, 255.9640476]),
    ("Bitter-W$_{INT1}$A$_{INT8}$", [0.503586429, 1.529837384, 135.626608]),
]

resnet_providers = ["BS1", "BS128"]
resnet_times_data = [
    ("PyTorch-Inductor", [2.372457981, 23.18973064]),
    ("ONNXRuntime", [2.7291, 84.9969]),
    ("TensorRT", [1.01499, 15.8538]),
    ("AMOS", [10.5099, 117.30035]),
    ("TensorIR", [1.344089898, 21.19485583]),
    ("Welder", [1.942134, 26.13936]),
    ("Bitter", [1.1762, 16.0981]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [1.1762, 16.0086471]),
    ("Bitter-W$_{MXFP8}$A$_{MXFP8}$", [1.921283197, 16.60434647]),
    ("Bitter-W$_{INT1}$A$_{INT4}$", [1.1762, 13.99871434]),
]

shufflenet_providers = ["BS1", "BS128"]
shufflenet_times_data = [
    ("PyTorch-Inductor", [3.560540676, 6.246321201]),
    ("ONNXRuntime", [1.6608, 10.0395]),
    ("TensorRT", [0.921546, 4.91514]),
    ("AMOS", [2.79608, 23.4862]),
    ("TensorIR", [0.354357554, 5.845679585]),
    ("Welder", [0.333824, 5.720064]),
    ("Bitter", [0.3483244, 4.837]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [0.3483244, 4.836]),
]

conformer_providers = ["BS1", "BS128"]
conformer_times_data = [
    ("PyTorch-Inductor", [5.971374512, 145.7871437]),
    ("ONNXRuntime", [5.961, 409.5517]),
    ("TensorRT", [1.77861, 151.155]),
    ("AMOS", [0, 0]),
    ("TensorIR", [0, 0]),
    ("Welder", [2.111488, 128.658524]),
    ("Bitter", [2.0192, 117.6661]),
    ("Bitter-W$_{INT4}$A$_{INT8}$", [1.783731184, 113.4327309]),
    ("Bitter-W$_{INT4}$A$_{INT4}$", [1.695257581, 111.486313]),
]

vit_providers = ["BS1", "BS128"]
vit_times_data = [
    ("PyTorch-Inductor", [2.6049757, 7.202584743]),
    ("ONNXRuntime", [2.3357, 19.111]),
    ("TensorRT", [0.470388, 6.76928]),
    ("AMOS", [0, 0]),
    ("TensorIR", [1.443533088, 9.853760724]),
    ("Welder", [1.068, 8.150656]),
    ("Bitter", [0.9576, 6.3083]),
    ("Bitter-W$_{FP8}$A$_{FP8}$", [0.788025605, 6.344933633]),
    ("Bitter-W$_{INT4}$A$_{INT4}$", [0.9576, 6.344933633]),
]