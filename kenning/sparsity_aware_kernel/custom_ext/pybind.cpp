#include <torch/extension.h>

#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {}

TORCH_LIBRARY(custom_ext, m)
{
    m.def("uncompress_weights(Tensor dequantized_weight, Tensor sparsity_metadata) -> Tensor");
    m.def("unquantize_weights(Tensor b_q_weight, Tensor b_gptq_qzeros, Tensor b_gptq_scales, Tensor b_g_idx, int bit) "
          "-> Tensor");
    m.def("reorder_metadata(Tensor(a!) sparsity_metadata) -> ()");
    m.def("compressed_gptq_gemm(Tensor a, Tensor b_q_weight, Tensor b_gptq_qzeros, Tensor b_gptq_scales, Tensor "
          "b_g_idx, Tensor sparsity_metadata, int bit, Tensor workspace, Tensor temp_dq) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_ext, CUDA, m)
{
    m.impl("uncompress_weights", &uncompress_weights);
    m.impl("unquantize_weights", &unquantize_weights);
    m.impl("reorder_metadata", &reorder_metadata);
    m.impl("compressed_gptq_gemm", &compressed_gptq_gemm);
}
