// vectorAdd_gpu.cu
#include <cuda_runtime.h>
#include "ternarus/vectorAdd.h"

__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

void vectorAddGPU(const float *d_A, const float *d_B, float *d_C, int N)
{
    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vectorAddKernel<<<numBlocks, blockSize>>>(d_A, d_B, d_C, N);

    // Synchronize to ensure the operation is completed
    cudaDeviceSynchronize();
}