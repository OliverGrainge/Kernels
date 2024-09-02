#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "ternarus/matMul.h"
#include "ternarus/ternMatMul.h"

// Function to bitpack the ternary matrix A
void packTernaryMatrix(const int8_t *A, uint32_t *packedA, int Arows, int Acols)
{
    int packedCols = (Acols + 15) / 16; // 16 ternary values per 32-bit word

    for (int i = 0; i < Arows; ++i)
    {
        for (int j = 0; j < packedCols; ++j)
        {
            uint32_t packed_val = 0;
            for (int k = 0; k < 16; ++k)
            {
                int col = j * 16 + k;
                if (col < Acols)
                {
                    int8_t a_val = A[i * Acols + col];
                    uint32_t bit_rep = (a_val == -1) ? 0b01 : (a_val == 1) ? 0b10
                                                                           : 0b00;
                    packed_val |= (bit_rep << (k * 2));
                }
            }
            packedA[i * packedCols + j] = packed_val;
        }
    }
}

int main()
{
    // Matrix dimensions
    const int Arows = 2024;
    const int Acols = 2024;
    const int Bcols = 2024;

    // Allocate host memory
    size_t sizeA_float = Arows * Acols * sizeof(float);
    size_t sizeA_int8 = Arows * Acols * sizeof(int8_t);
    size_t sizeB = Acols * Bcols * sizeof(float);
    size_t sizeC = Arows * Bcols * sizeof(float);

    float *A_float = new float[Arows * Acols];
    int8_t *A_int8 = new int8_t[Arows * Acols];
    float *B = new float[Acols * Bcols];
    float *C_gpu_tern = new float[Arows * Bcols];
    float *C_gpu_float = new float[Arows * Bcols];
    float *C_gpu_bitpacked = new float[Arows * Bcols];

    // Initialize input matrices with random values for float matrix multiplication
    for (int i = 0; i < Arows * Acols; ++i)
    {
        A_float[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < Acols * Bcols; ++i)
    {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Initialize input matrices with random values for ternary matrix multiplication
    for (int i = 0; i < Arows * Acols; ++i)
    {
        int r = rand() % 3;
        A_int8[i] = (r == 0) ? -1 : (r == 1) ? 0
                                             : 1;
    }

    // Allocate device memory
    float *d_A_float, *d_B, *d_C;
    int8_t *d_A_int8;
    uint32_t *d_packedA;

    cudaMalloc((void **)&d_A_float, sizeA_float);
    cudaMalloc((void **)&d_A_int8, sizeA_int8);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    // Bitpacking
    int packedAcols = (Acols + 15) / 16;
    size_t sizePackedA = Arows * packedAcols * sizeof(uint32_t);
    uint32_t *packedA = new uint32_t[Arows * packedAcols];
    packTernaryMatrix(A_int8, packedA, Arows, Acols);

    // Allocate device memory for bitpacked A
    cudaMalloc((void **)&d_packedA, sizePackedA);
    cudaMemcpy(d_packedA, packedA, sizePackedA, cudaMemcpyHostToDevice);

    // Copy data from host to device
    cudaMemcpy(d_A_float, A_float, sizeA_float, cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_int8, A_int8, sizeA_int8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    // Perform matrix multiplication on GPU with floating-point data
    auto start_gpu_float = std::chrono::high_resolution_clock::now();
    matMulGPU(d_A_float, d_B, d_C, Arows, Acols, Bcols); // No data transfer to host within this function
    cudaDeviceSynchronize();                             // Ensure GPU operations are complete
    auto end_gpu_float = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time_float = end_gpu_float - start_gpu_float;

    // Copy result from device to host (optional if you want to verify)
    cudaMemcpy(C_gpu_float, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Perform matrix multiplication on GPU with ternary data
    auto start_gpu_tern = std::chrono::high_resolution_clock::now();
    ternMatMulGPU(d_A_int8, d_B, d_C, Arows, Acols, Bcols); // No data transfer to host within this function
    cudaDeviceSynchronize();                                // Ensure GPU operations are complete
    auto end_gpu_tern = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time_tern = end_gpu_tern - start_gpu_tern;

    // Copy result from device to host (optional if you want to verify)
    cudaMemcpy(C_gpu_tern, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Perform matrix multiplication on GPU with bitpacked ternary data
    auto start_gpu_bitpacked = std::chrono::high_resolution_clock::now();
    ternMatMulBitpackedGPU(d_packedA, d_B, d_C, Arows, Acols, Bcols); // No data transfer to host within this function
    cudaDeviceSynchronize();                                          // Ensure GPU operations are complete
    auto end_gpu_bitpacked = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time_bitpacked = end_gpu_bitpacked - start_gpu_bitpacked;

    // Copy result from device to host (optional if you want to verify)
    cudaMemcpy(C_gpu_bitpacked, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Output the execution times
    std::cout << "GPU Time (Floating-Point MatMul): " << gpu_time_float.count() << " seconds" << std::endl;
    std::cout << "GPU Time (Ternary MatMul): " << gpu_time_tern.count() << " seconds" << std::endl;
    std::cout << "GPU Time (Bitpacked Ternary MatMul): " << gpu_time_bitpacked.count() << " seconds" << std::endl;

    // Clean up
    delete[] A_float;
    delete[] A_int8;
    delete[] B;
    delete[] C_gpu_float;
    delete[] C_gpu_tern;
    delete[] C_gpu_bitpacked;
    delete[] packedA;
    cudaFree(d_A_float);
    cudaFree(d_A_int8);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_packedA);

    return 0;
}
