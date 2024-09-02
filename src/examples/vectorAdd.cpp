#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "ternarus/vectorAdd.h"

int main()
{
    const int N = 1 << 20; // 1 Million elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *A = new float[N];
    float *B = new float[N];
    float *C_cpu = new float[N];
    float *C_gpu = new float[N];

    // Initialize input vectors with random values
    for (int i = 0; i < N; ++i)
    {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Perform vector addition on CPU
    auto start_cpu = std::chrono::high_resolution_clock::now();
    vectorAddCPU(A, B, C_cpu, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_time = end_cpu - start_cpu;

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy data from host to device
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Perform vector addition on GPU
    auto start_gpu = std::chrono::high_resolution_clock::now();
    vectorAddGPU(d_A, d_B, d_C, N); // Note: No data transfer to host within this function
    cudaDeviceSynchronize();        // Ensure GPU operations are complete
    auto end_gpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time = end_gpu - start_gpu;

    // Copy result from device to host
    cudaMemcpy(C_gpu, d_C, size, cudaMemcpyDeviceToHost);

    // Verify results
    bool result_correct = true;
    for (int i = 0; i < N; ++i)
    {
        if (fabs(C_cpu[i] - C_gpu[i]) > 1e-5)
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
