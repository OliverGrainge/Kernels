#include <cuda_runtime.h>
#include "ternarus/matMul.h"
#include <cstdint>

// Kernel for float
__global__ void matMulKernelFloat(const float *A, const float *B, float *C, int Arows, int Acols, int Bcols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float temp_sum = 0.0f;
    if ((row < Arows) && (col < Bcols))
    {
        for (int k = 0; k < Acols; k++)
        {
            temp_sum += A[row * Acols + k] * B[k * Bcols + col];
        }
        C[row * Bcols + col] = temp_sum;
    }
}

// Wrapper for float
void matMulGPUFloat(const float *A, const float *B, float *C, int Arows, int Acols, int Bcols)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((Bcols + blockDim.x - 1) / blockDim.x, (Arows + blockDim.y - 1) / blockDim.y);

    matMulKernelFloat<<<gridDim, blockDim>>>(A, B, C, Arows, Acols, Bcols);
    cudaDeviceSynchronize();
}

// Kernel for uint8_t
__global__ void matMulKernelUint8(const uint8_t *A, const uint8_t *B, uint8_t *C, int Arows, int Acols, int Bcols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    int temp_sum = 0;
    if ((row < Arows) && (col < Bcols))
    {
        for (int k = 0; k < Acols; k++)
        {
            temp_sum += A[row * Acols + k] * B[k * Bcols + col];
        }
        C[row * Bcols + col] = (uint8_t)min(temp_sum, 255);
    }
}

// Wrapper for uint8_t
void matMulGPUUint8(const uint8_t *A, const uint8_t *B, uint8_t *C, int Arows, int Acols, int Bcols)
{
    dim3 blockDim(16, 16);
    dim3 gridDim((Bcols + blockDim.x - 1) / blockDim.x, (Arows + blockDim.y - 1) / blockDim.y);

    matMulKernelUint8<<<gridDim, blockDim>>>(A, B, C, Arows, Acols, Bcols);
    cudaDeviceSynchronize();
}
