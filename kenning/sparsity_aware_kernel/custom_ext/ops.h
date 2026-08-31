#pragma once

#include <torch/extension.h>

torch::Tensor uncompress_weights(torch::Tensor dequantized_weight, torch::Tensor sparsity_metadata);

torch::Tensor unquantize_weights(
    torch::Tensor b_q_weight,
    torch::Tensor b_gptq_qzeros,
    torch::Tensor b_gptq_scales,
    torch::Tensor b_g_idx,
    const int64_t bit);

torch::Tensor compressed_gptq_gemm(
    torch::Tensor a,
    torch::Tensor b_q_weight,
    torch::Tensor b_gptq_qzeros,
    torch::Tensor b_gptq_scales,
    torch::Tensor b_g_idx,
    torch::Tensor sparsity_metadata,
    int64_t bit,
    torch::Tensor workspace,
    torch::Tensor temp_dq);

void reorder_metadata(torch::Tensor sparsity_metadata);
