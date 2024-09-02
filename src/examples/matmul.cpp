#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "ternarus/matMul.h"

int main()
{
    const int Arows = 1024;
    const int Acols = 512;
    const int Bcols = 1024;
    size_t sizeA = Arows * Acols * sizeof(float);
    size_t sizeB = Acols * Bcols * sizeof(float);
    size_t sizeC = Arows * Bcols * sizeof(float);

    // Allocate host memory
    float *A = new float[Arows * Acols];
    float *B = new float[Acols * Bcols];
    float *C_cpu = new float[Arows * Bcols];
    float *C_gpu = new float[Arows * Bcols];

    // Initialize input matrices with random values
    for (int i = 0; i < Arows * Acols; ++i)
    {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < Acols * Bcols; ++i)
    {
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Perform matrix multiplication on CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matMulCPU(A, B, C_cpu, Arows, Acols, Bcols);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, sizeA);
    cudaMalloc((void **)&d_B, sizeB);
    cudaMalloc((void **)&d_C, sizeC);

    // Copy data from host to device
    cudaMemcpy(d_A, A, sizeA, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, sizeB, cudaMemcpyHostToDevice);

    // Perform matrix multiplication on GPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    matMulGPU(d_A, d_B, d_C, Arows, Acols, Bcols); // Note: No data transfer to host within this function
    cudaDeviceSynchronize();                       // Ensure GPU operations are complete
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // Copy result from device to host
    cudaMemcpy(C_gpu, d_C, sizeC, cudaMemcpyDeviceToHost);

    // Verify results
    bool result_correct = true;
    for (int i = 0; i < Arows * Bcols; ++i)
    {
        if (fabs(C_cpu[i] - C_gpu[i]) > 1e-4)
        {
            result_correct = false;
            std::cout << "Mismatch at index " << i << ": CPU result = "
                      << C_cpu[i] << ", GPU result = " << C_gpu[i] << std::endl;
            break;
        }
    }

    if (result_correct)
    {
        std::cout << "Results are correct!" << std::endl;
        std::cout << "CPU Time: " << cpu_time.count() << " seconds" << std::endl;
        std::cout << "GPU Time: " << gpu_time.count() << " seconds" << std::endl;
    }
    else
    {
        std::cout << "Results are incorrect!" << std::endl;
    }

    // Clean up
    delete[] A;
    delete[] B;
    delete[] C_cpu;
    delete[] C_gpu;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
