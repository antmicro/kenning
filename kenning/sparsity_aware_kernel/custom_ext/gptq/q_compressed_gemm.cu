/*
 * Copyright (c) 2024 Antmicro <www.antmicro.com>
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm_sparse.h>
#include <cutlass/util/host_reorder.h>
#include <cutlass/util/host_tensor.h>
#include <cutlass/util/host_uncompress.h>
#include <cutlass/util/reference/host/gemm.h>

#include "third_party/matrix_view.cuh"
#include "utils.hpp"

#define BLOCK_KN_SIZE 64
#define BUFFER_SIZE 32
#define GRID_K_SIZE 4
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))
#define GPTQ_GEMM_DEBUG 0
#define VALUES_PER_METADATA 8
// Sparse matrix passed to the CUTLASS kernel has to have
// both dimensions divisible by 8
#define CUTLASS_GEMM_M 8
#define COMPRESSION_RATIO 2
#define VALUES_PER_QUANTIZED_ELEMENT 8 // It means that in 32 bits there are 8 quantized values
#define STAGES 2

using SparseInputLayout = cutlass::layout::ColumnMajor;
using DenseInputLayout = cutlass::layout::ColumnMajor;
using InputElement = cutlass::half_t;

using OutputLayout = cutlass::layout::RowMajor;
using OutputElement = cutlass::half_t;

using MetadataLayout = cutlass::layout::RowMajor;
using MetadataElement = uint16_t;
using ReorderedInputELayout = cutlass::layout::ColumnMajorInterleaved<2>;

using AccumulatorElement = float;

using MMAOp = cutlass::arch::OpClassTensorOp;

using SmArch = cutlass::arch::Sm80;

/**
 * Kernel that performs sparse matrix-vector multiplication
 *
 * @param vec Input vector
 * @param matrix Sparse quantized matrix
 * @param sparsity_metadata Sparsity metadata
 * @param scales Scales for each group
 * @param zeros Zeros for each group
 * @param g_idx Group indices
 * @param output Output matrix
 * @param m Height of the output matrix
 * @param n Width of the output matrix
 * @param k Width of the input matrix
 * @param group Number of groups
 */
__global__ void sparse_mv_mult(
    const half* __restrict__ vec,
    const uint32_t* __restrict__ matrix,
    const half* __restrict__ scales,
    const uint32_t* __restrict__ zeros,
    const int* __restrict__ g_idx,
    half* __restrict__ output,
    const int m,
    const int n,
    const int k,
    const int group_num,
    const cutlass::TensorRef<MetadataElement, ReorderedInputELayout> tensor_e_reordered
)
{
  const int batch_size = n;
  // Each thread is responsible for one row of the matrix
  const int output_offset = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;

  if (output_offset >= m) {
    return;
  }

  // Obtaining the first element of the row
  const int matrix_height = m;
  const int matrix_width = k / COMPRESSION_RATIO;
  const int quantized_matrix_width = k / COMPRESSION_RATIO / VALUES_PER_QUANTIZED_ELEMENT;
  const int initial_matrix_row = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;
  const int initial_matrix_column = blockIdx.y * (quantized_matrix_width / GRID_K_SIZE);
  const int matrix_offset = initial_matrix_column * matrix_height + initial_matrix_row;

  // Obtaining the first element of the metadata
  const int metadata_height = m;
  const int metadata_width = k / COMPRESSION_RATIO / VALUES_PER_METADATA;
  const int initial_metadata_row = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;
  const int initial_metadata_column = blockIdx.y * (metadata_width / GRID_K_SIZE);

  const int group_size = k / COMPRESSION_RATIO / group_num;

  // Obtaining the first element of scales
  const int scales_height = m;
  const int scales_width = group_num;
  const int initial_scales_row = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;
  const int initial_scales_column = blockIdx.y * (scales_width / GRID_K_SIZE);
  const int scales_offset = initial_scales_column * scales_height + initial_scales_row;

  // Obtaining the first element of zero
  const int zeros_height = m / VALUES_PER_QUANTIZED_ELEMENT;
  const int zeros_width = group_num;
  const int initial_zeros_row = (BLOCK_KN_SIZE * blockIdx.x + threadIdx.x) / VALUES_PER_QUANTIZED_ELEMENT;
  const int initial_zeros_column = blockIdx.y * (zeros_width / GRID_K_SIZE);
  const int zeros_offset = initial_zeros_column * zeros_height + initial_zeros_row;

  // STAGES x num of tokens x BUFFER_SIZE * 2
  __shared__ float vector_block[STAGES][CUTLASS_GEMM_M - 1][BUFFER_SIZE * COMPRESSION_RATIO];
  float results[CUTLASS_GEMM_M - 1] = {0.0f};

  for (int i = 0; i < matrix_width / BUFFER_SIZE / GRID_K_SIZE; i += 1) {
    // writing vector information using memory coalescing
    if (threadIdx.x < BUFFER_SIZE * COMPRESSION_RATIO) {
      for (int j = 0; j < batch_size; j++) {
        const int vector_column = j;
        const int vector_height = k;
        const int vector_row = i * BUFFER_SIZE * COMPRESSION_RATIO + blockIdx.y * (vector_height / GRID_K_SIZE);
        vector_block[i % STAGES][j][threadIdx.x] = __half2float(vec[vector_column * vector_height + vector_row + threadIdx.x]);
      }
    }

    __syncthreads();

    // Assuming that BUFFER_SIZE < group_size and BUFFER_SIZE | group size
    const int scales_column = i * BUFFER_SIZE / group_size;
    const int scales_index = scales_offset + scales_column * scales_height;
    const float scale = __half2float(scales[scales_index]);

    const int zeros_column = i * BUFFER_SIZE / group_size;
    const int zeros_index = (zeros_offset + zeros_column * zeros_height);
    const uint32_t zero = (zeros[zeros_index] >> (4 * (threadIdx.x % 8))) & 0xF;

    for (int b = 0; b < batch_size; b++) {

      #pragma unroll
      for (int j = 0; j < BUFFER_SIZE / VALUES_PER_METADATA; j += 1) {
        const int matrix_element = i * BUFFER_SIZE + j * VALUES_PER_METADATA;
        const int matrix_column = matrix_element / VALUES_PER_QUANTIZED_ELEMENT;
        const int matrix_index = (matrix_offset + matrix_column * matrix_height);
        const int matrix_block_value = matrix[matrix_index];

        const int m = initial_metadata_row;
        const int k = initial_metadata_column + (i * (BUFFER_SIZE / VALUES_PER_METADATA) + j);

        const int group = 32;
        const int interweave = 4;

        int dest_row = m / group * group + (m % 8) * interweave + (m % group) / 8;
        int dest_col = k;

        if (((dest_row % 2) == 0) && ((dest_col % 2) == 1)) {
            ++dest_row;
            --dest_col;
        } else if (((dest_row % 2) == 1) && ((dest_col % 2) == 0)) {
            --dest_row;
            ++dest_col;
        }

        const uint16_t metadata_block = tensor_e_reordered.at({dest_row, dest_col});

        #pragma unroll
        for (int l = 0; l < VALUES_PER_METADATA; l++) {
          const int matrix_value = (matrix_block_value >> (4 * (l % 8))) & 0xF;
          const float matrix_subtracted_value = __int2float_rn(matrix_value - zero);
          const float matrix_unquantized_value = matrix_subtracted_value * scale;

          const int metadata_index = (metadata_block >> (l * 2)) & 0b11;

          const int vector_block_index = j * VALUES_PER_METADATA * 2 + (l / 2) * 4;
          const float vector_value = vector_block[i % STAGES][b][vector_block_index + metadata_index];

          results[b] += vector_value * matrix_unquantized_value;
        }
      }
    }
  }

  for (int i = 0; i < batch_size; i++) {
    const int output_row = output_offset;
    const int output_column = i;
    const int output_index = output_column + output_row * batch_size;
    atomicAdd(&output[output_index],  __float2half(results[i]));
  }
}


/**
 * Kernel that reconstruct compressed GPTQ weights
 *
 * @param w Compressed GPTQ weights
 * @param w_scales Scales for each group
 * @param w_zeros Zeros for each group
 * @param g_idx Group indices
 * @param height Height of the output matrix
 * @param width Width of the output matrix
 * @param group Number of groups
 * @param out Output matrix where the weights will be reconstructed
 */

__global__ void reconstruct_compressed_gptq_kernel(
    const uint32_t *__restrict__ w, const half *__restrict__ w_scales,
    const uint32_t *__restrict__ w_zeros, const int *__restrict__ g_idx,
    const int height, const int width, const int group,
    half *__restrict__ out) {
  const auto bit = 4;

  // Start of block
  int column = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;
  int row = blockIdx.y * 32 / bit;
  if (column >= width)
    return;

  // Views
  vllm::gptq::MatrixView_half_rw out_(out, height, width);
  vllm::gptq::MatrixView_half w_scales_(w_scales, group, width);
  vllm::gptq::MatrixView_q4_row w_zeros_(w_zeros, group, width);

  uint32_t w_read = w[blockIdx.y * width + column];
  half *out_ptr = out_.item_ptr(row, column);

#pragma unroll
  for (int s = 0; s < 32; s += bit) {
    int group = g_idx[row + s / bit];
    half w_scale = w_scales_.item(group, column);
    uint32_t w_zero = w_zeros_.item(group, column);
    half w_item =
        __hmul(__int2half_rn((int)((w_read >> s) & ((1 << bit) - 1)) - w_zero),
               w_scale);
    *out_ptr = w_item;
    out_ptr += out_.width;
  }
}

/**
 * Reconstruct compressed GPTQ weights\
 *
 * Wrapper for the kernel that reconstructs compressed GPTQ weights
 *
 * @param b_q_weight Compressed GPTQ weights
 * @param b_gptq_qzeros Zeros for each group
 * @param b_gptq_scales Scales for each group
 * @param b_g_idx Group indices
 * @param out Output matrix where the weights will be reconstructed
 * @param height Height of the output matrix
 * @param width Width of the output matrix
 * @param groups Number of groups
 * @param bit Bit precision that was used for quantization
 */
void reconstruct_compressed_gptq(const uint32_t *b_q_weight,
                                 const uint32_t *b_gptq_qzeros,
                                 const half *b_gptq_scales, const int *b_g_idx,
                                 half *out, const int height, const int width, const int groups,
                                 const int bit) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;

  gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);
  gridDim.y = DIVIDE(height, 32 / bit);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  reconstruct_compressed_gptq_kernel<<<gridDim, blockDim, 0, stream>>>(
      b_q_weight, b_gptq_scales, b_gptq_qzeros, b_g_idx, height, width, groups,
      out);
}

/**
 * Pytorch binding for reconstructing compressed GPTQ weights.
 *
 * @param b_q_weight Compressed GPTQ weights
 * @param b_gptq_qzeros Zeros for each group
 * @param b_gptq_scales Scales for each group
 * @param b_g_idx Group indices
 * @param bit Bit precision that was used for quantization
 *
 * @return Dequantized pytorch tensor
 */
torch::Tensor unquantize_weights(torch::Tensor b_q_weight,
                                torch::Tensor b_gptq_qzeros,
                                torch::Tensor b_gptq_scales,
                                torch::Tensor b_g_idx, const int bit) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(b_q_weight));
  auto temp_dq_options =
      torch::TensorOptions().dtype(torch::kFloat16).device(b_q_weight.device());

  // Dequantized weight matrix
  at::Tensor temp_dq = torch::empty(
      {b_q_weight.size(0) * 32 / bit, b_q_weight.size(1)}, temp_dq_options);

  const int m = temp_dq.size(1);
  const int k = temp_dq.size(0);

#if GPTQ_GEMM_DEBUG
  std::cout << "Weights matrix size: " << b_q_weight.size(0) << ", "
            << b_q_weight.size(1) << std::endl;
  std::cout << "temp_dq matrix size: " << temp_dq.size(0) << ", "
            << temp_dq.size(1) << std::endl;
  std::cout << "m: " << m << ", k: " << k << std::endl;
#endif

  auto b_q_weight_ptr = (const uint32_t *)b_q_weight.data_ptr();
  auto b_gptq_qzeros_ptr = (const uint32_t *)b_gptq_qzeros.data_ptr();
  auto b_gptq_scales_ptr = (const half *)b_gptq_scales.data_ptr();
  auto b_g_idx_ptr = (const int *)b_g_idx.data_ptr();

  auto temp_dq_ptr = (half *)temp_dq.data_ptr();

  const auto groups = b_gptq_qzeros.size(0);

  reconstruct_compressed_gptq(b_q_weight_ptr, b_gptq_qzeros_ptr,
                              b_gptq_scales_ptr, b_g_idx_ptr, temp_dq_ptr, k, m,
                              groups, bit);

  return temp_dq;
}

/**
 * Pytorch binding for uncompressing weights into a sparse format
 *
 * @param dequantized_weight Dequantized, compressed weight matrix
 * @param sparsity_metadata Sparsity metadata
 *
 * @return Uncompressed weight matrix
 */

torch::Tensor uncompress_weights(torch::Tensor dequantized_weight,
                                 torch::Tensor sparsity_metadata) {
  const int m = dequantized_weight.size(1);
  const int k = dequantized_weight.size(0);

  // Weight matrix
  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>
      sparse_matrix_host({m, k});
  CHECK_CUDA(cudaMemcpy((void *)sparse_matrix_host.host_data(),
                        (void *)dequantized_weight.data_ptr(),
                        m * k * sizeof(cutlass::half_t),
                        cudaMemcpyDeviceToHost));

  // metadata matrix
  cutlass::HostTensor<uint16_t, cutlass::layout::RowMajor> metadata_host(
      {m, k / VALUES_PER_METADATA});
  CHECK_CUDA(cudaMemcpy((void *)metadata_host.host_data(),
                        (void *)sparsity_metadata.data_ptr(),
                        m * k / VALUES_PER_METADATA * sizeof(uint16_t), cudaMemcpyDeviceToHost));

  cutlass::HostTensor<cutlass::half_t, cutlass::layout::ColumnMajor>
      uncompressed_tensor_a({m, k * COMPRESSION_RATIO});

  cutlass::uncompress(uncompressed_tensor_a.host_ref(),
                      sparse_matrix_host.host_ref(), metadata_host.host_ref(),
                      m, k * COMPRESSION_RATIO);

  auto options =
      torch::TensorOptions().dtype(torch::kFloat16).device(torch::kCUDA);
  torch::Tensor re = torch::empty({m, k * COMPRESSION_RATIO}, options);
  CHECK_CUDA(cudaMemcpy(
      (void *)re.data_ptr(), (void *)uncompressed_tensor_a.host_data(),
      m * k * COMPRESSION_RATIO * sizeof(cutlass::half_t), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaDeviceSynchronize());
  return re;
}

/**
 * Pytorch binding for reordering metadata.
 *
 * Reorders metadata into a format that is required by CUTLASS sparse
 * operations. The operation is done in-place.
 *
 * @param sparsity_metadata Sparsity metadata
 */

void reorder_metadata(torch::Tensor sparsity_metadata) {
  const int m = sparsity_metadata.size(0);
  const int k = sparsity_metadata.size(1);
  uint16_t *sparsity_metadata_ptr = (uint16_t *)sparsity_metadata.data_ptr();

  int16_t *host_sparsity_metadata_ptr =
      (int16_t *)malloc(m * k * sizeof(MetadataElement));
  CHECK_CUDA(cudaMemcpy(
      (void *)host_sparsity_metadata_ptr, (void *)sparsity_metadata_ptr,
      m * k * sizeof(MetadataElement), cudaMemcpyDeviceToHost));

  cutlass::HostTensor<MetadataElement, ReorderedInputELayout>
      tensor_e_reordered(cutlass::make_Coord(m, k));

  cutlass::TensorRef<MetadataElement, MetadataLayout> tensor_e(
      (MetadataElement *)host_sparsity_metadata_ptr, k);
  cutlass::reorder_meta(tensor_e_reordered.host_ref(), tensor_e,
                        {m, 0, k}); // Second dimension is not used

  CHECK_CUDA(cudaMemcpy(
      (void *)sparsity_metadata_ptr, (void *)tensor_e_reordered.host_data(),
      m * k * sizeof(MetadataElement), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaDeviceSynchronize());
}

/**
 * Pytorch binding for performing sparse GPTQ GEMM operation.
 *
 * @param a Input matrix
 * @param b_q_weight Compressed GPTQ weights
 * @param b_gptq_qzeros Zeros for each group
 * @param b_gptq_scales Scales for each group
 * @param b_g_idx Group indices
 * @param sparsity_metadata Sparsity metadata
 * @param bit Bit precision that was used for quantization
 *
 * @return GEMM result, that is a product of `b_q_weight` weights and `a` matrix
 */
torch::Tensor compressed_gptq_gemm(const torch::Tensor a,
                                   const torch::Tensor b_q_weight,
                                   const torch::Tensor b_gptq_qzeros,
                                   const torch::Tensor b_gptq_scales,
                                   const torch::Tensor b_g_idx,
                                   const torch::Tensor sparsity_metadata,
                                   const int bit) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto c_options =
      torch::TensorOptions().dtype(torch::kFloat16).device(a.device());

  // Output matrix
  at::Tensor c = torch::zeros({b_q_weight.size(1), a.size(0)}, c_options);

  const int m = c.size(0);
  const int n = c.size(1);
  const int k = a.size(1);

#if GPTQ_GEMM_DEBUG
  // Important links:
  // https://github.com/NVIDIA/cutlass/blob/main/media/docs/layout.md
  // https://github.com/NVIDIA/cutlass/blob/main/media/docs/utilities.md
  std::cout << "A matrix size: " << a.size(0) << ", " << a.size(1) << std::endl;
  std::cout << "B matrix size: " << b_q_weight.size(0) << ", "
            << b_q_weight.size(1) << std::endl;
  std::cout << "C matrix size: " << c.size(0) << ", " << c.size(1) << std::endl;
  std::cout << "sparsity metadata size: " << sparsity_metadata.size(0) << ", "
            << sparsity_metadata.size(1) << std::endl;
  std::cout << "m: " << m << ", n: " << n << ", k: " << k << std::endl;
#endif

  auto a_ptr = (const half *)a.data_ptr();
  auto b_q_weight_ptr = (const uint32_t *)b_q_weight.data_ptr();
  auto b_gptq_qzeros_ptr = (const uint32_t *)b_gptq_qzeros.data_ptr();
  auto b_gptq_scales_ptr = (const half *)b_gptq_scales.data_ptr();
  auto b_g_idx_ptr = (const int *)b_g_idx.data_ptr();
  uint16_t *sparsity_metadata_ptr = (uint16_t *)sparsity_metadata.data_ptr();

  auto c_ptr = (half *)c.data_ptr();

  const auto groups = b_gptq_qzeros.size(0);

  // Stride is set to 2 * m because constructor of ColumnMajorInterleaved<2>
  // calculates it as extent.row() * kInterleave
  auto reordered_metadata_layout = ReorderedInputELayout(2 * m);
  cutlass::TensorRef<MetadataElement, ReorderedInputELayout> tensor_e_reordered(
      (MetadataElement *)sparsity_metadata_ptr, reordered_metadata_layout);

  // If the input is too small to be run with cutlass, run the custom kernel
  if (n < CUTLASS_GEMM_M) {
    dim3 blockDim, gridDim;
    blockDim.x = BLOCK_KN_SIZE;
    // blockDim.x = 1;
    blockDim.y = 1;
    blockDim.z = 1;
    gridDim.x = DIVIDE(m, BLOCK_KN_SIZE);
    // gridDim.x = 1;
    gridDim.y = GRID_K_SIZE;
    gridDim.z = 1;
    // Special case for 1D input
    // Every thread is responsible for one row
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    sparse_mv_mult<<<gridDim, blockDim, 0, stream>>>(
      a_ptr, b_q_weight_ptr,
      b_gptq_scales_ptr, b_gptq_qzeros_ptr, b_g_idx_ptr,
      c_ptr, m, n, k, groups, tensor_e_reordered);
    return c;
  }

  auto temp_dq_options =
  torch::TensorOptions().dtype(a.dtype()).device(a.device());

  // Dequantized weight matrix
  at::Tensor temp_dq = torch::empty(
      {b_q_weight.size(0) * 32 / bit, b_q_weight.size(1)}, temp_dq_options);

#if GPTQ_GEMM_DEBUG
  std::cout << "temp_dq matrix size: " << temp_dq.size(0) << ", "
          << temp_dq.size(1) << std::endl;
#endif

  auto temp_dq_ptr = (half *)temp_dq.data_ptr();

  reconstruct_compressed_gptq(b_q_weight_ptr, b_gptq_qzeros_ptr,
                            b_gptq_scales_ptr, b_g_idx_ptr, temp_dq_ptr,
                            k / COMPRESSION_RATIO, m, groups, bit);

  // Column-major sparse matrix
  cutlass::TensorRef<InputElement, SparseInputLayout> sparse_matrix(
      (InputElement *)temp_dq_ptr, m);

  // Column-major input
  cutlass::TensorRef<InputElement, DenseInputLayout> dense_matrix(
      (InputElement *)a_ptr, k);

  // Row-major output
  cutlass::TensorRef<OutputElement, OutputLayout> output_matrix(
      (OutputElement *)c_ptr, n);

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 64>;
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 64>;
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 32>;
  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>; // <- ??

  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      OutputElement, 128 / cutlass::sizeof_bits<OutputElement>::value,
      AccumulatorElement, AccumulatorElement>;

  const int NumStages = 3;

  using Gemm = cutlass::gemm::device::SparseGemm<
      InputElement, SparseInputLayout, InputElement, DenseInputLayout,
      OutputElement, OutputLayout, AccumulatorElement,
      cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;

  AccumulatorElement alpha = AccumulatorElement(1);
  AccumulatorElement beta = AccumulatorElement(0);

  int split_k_slices = 1;

  Gemm::Arguments arguments{{m, n, k},     sparse_matrix, dense_matrix,
                            output_matrix, output_matrix, tensor_e_reordered,
                            {alpha, beta}, split_k_slices};

  size_t workspace_size = Gemm::get_workspace_size(arguments);
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  Gemm gemm_op;
  CHECK_CUTLASS(gemm_op.can_implement(arguments));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  CHECK_CUTLASS(gemm_op.initialize(arguments, workspace.get(), stream));
  CHECK_CUTLASS(gemm_op(stream));

  return c;
}
