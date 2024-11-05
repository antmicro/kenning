#include <torch/extension.h>

#include "ops.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("uncompress_weights", &uncompress_weights, "Test for metadata");
    m.def("unquantize_weights", &unquantize_weights, "Test for metadata");
    m.def("reorder_metadata", &reorder_metadata, "Preprocessed metadata to be used by sparse CUTLASS kernel");
    m.def("compressed_gptq_gemm", &compressed_gptq_gemm, "Quantized Compressed GEMM for GPTQ");
}
