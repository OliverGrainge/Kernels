#include <cuda_runtime.h>
#include "ternarus/matMul.h"

__global__ void matMulKernel(const float *A, const float *B, float *C, int Arows, int Acols, int Bcols)
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
        C[row * Bcols + col] = temp_sum; // Use Bcols instead of Acols
    }
}

void matMulGPU(const float *A, const float *B, float *C, int Arows, int Acols, int Bcols)
{
    // Define grid and block dimensions
    dim3 blockDim(16, 16); // Adjust block size for your GPU
    dim3 gridDim((Bcols + blockDim.x - 1) / blockDim.x, (Arows + blockDim.y - 1) / blockDim.y);

    // Launch the kernel
    matMulKernel<<<gridDim, blockDim>>>(A, B, C, Arows, Acols, Bcols);

    // Optionally, you might want to synchronize to ensure kernel completion before continuing
    cudaDeviceSynchronize();
}
