#include <cuda_runtime.h>
#include <cutlass/cutlass.h>

#define CHECK_CUDA(func)                                                                                          \
    {                                                                                                             \
        cudaError_t status = (func);                                                                              \
        if (status != cudaSuccess)                                                                                \
        {                                                                                                         \
            std::cerr << "CUDA API failed at line: " << __LINE__ << " with error: " << cudaGetErrorString(status) \
                      << "(" << status << ")" << std::endl;                                                       \
            exit(EXIT_FAILURE);                                                                                   \
        }                                                                                                         \
    }

#define CHECK_CUTLASS(func)                                                             \
    {                                                                                   \
        cutlass::Status error = (func);                                                 \
        if (error != cutlass::Status::kSuccess)                                         \
        {                                                                               \
            std::cerr << "CUTLASS API failed at line: " << __LINE__                     \
                      << " with error: " << cutlassGetStatusString(error) << std::endl; \
            exit(EXIT_FAILURE);                                                         \
        }                                                                               \
    }
