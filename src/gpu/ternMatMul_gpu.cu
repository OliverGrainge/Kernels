#include <cuda_runtime.h>
#include <cstdint>
#include "ternarus/ternMatMul.h"

__global__ void ternMatMulKernel(const int8_t *A, const float *B, float *C, int Arows, int Acols, int Bcols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float temp_sum = 0.0f;
    if ((row < Arows) && (col < Bcols))
    {
        for (int k = 0; k < Acols; k++)
        {
            int8_t a_val = A[row * Acols + k];
            temp_sum += a_val * B[k * Bcols + col]; // a_val is -1, 0, or 1
        }
        C[row * Bcols + col] = temp_sum;
    }
}

// ternMatMulBitpacked.cu
#include <cuda_runtime.h>
#include <cstdint>
#include "ternarus/ternMatMul.h"

__global__ void ternMatMulBitpackedKernel(const uint32_t *packedA, const float *B, float *C, int Arows, int packedAcols, int Bcols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float temp_sum = 0.0f;
    if ((row < Arows) && (col < Bcols))
    {
        for (int j = 0; j < packedAcols; ++j)
        {
            uint32_t packed_val = packedA[row * packedAcols + j];
            for (int k = 0; k < 16; ++k)
            {
                uint32_t bit_rep = (packed_val >> (k * 2)) & 0b11;
                int8_t a_val = (bit_rep == 0b01) ? -1 : (bit_rep == 0b10) ? 1
                                                                          : 0;
                int col_idx = j * 16 + k;
                if (col_idx < Bcols)
                {
                    temp_sum += a_val * B[col_idx * Bcols + col];
                }
            }
        }
        C[row * Bcols + col] = temp_sum;
    }
}

extern "C" void ternMatMulBitpackedGPU(const uint32_t *packedA, const float *B, float *C, int Arows, int Acols, int Bcols)
{
    int packedAcols = (Acols + 15) / 16;
    dim3 blockDim(16, 16);
    dim3 gridDim((Bcols + blockDim.x - 1) / blockDim.x, (Arows + blockDim.y - 1) / blockDim.y);

    ternMatMulBitpackedKernel<<<gridDim, blockDim>>>(packedA, B, C, Arows, packedAcols, Bcols);

    cudaDeviceSynchronize();
}

void ternMatMulGPU(const int8_t *A, const float *B, float *C, int Arows, int Acols, int Bcols)
{
    // Define grid and block dimensions
    dim3 blockDim(16, 16); // Adjust block size for your GPU
    dim3 gridDim((Bcols + blockDim.x - 1) / blockDim.x, (Arows + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    ternMatMulKernel<<<gridDim, blockDim>>>(A, B, C, Arows, Acols, Bcols);

    // Optionally, you might want to synchronize to ensure kernel completion before continuing
    cudaDeviceSynchronize();
}
