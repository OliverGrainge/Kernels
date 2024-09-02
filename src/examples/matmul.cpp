#include <iostream>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cstdint>
#include "ternarus/matMul.h" // Make sure to replace this with the actual name of your header file

int main()
{
    const int Arows = 1024;
    const int Acols = 1024;
    const int Bcols = 1024;

    // Allocate and initialize host memory for float matrices
    float *A_float = new float[Arows * Acols];
    float *B_float = new float[Acols * Bcols];
    float *C_float_cpu = new float[Arows * Bcols];
    float *C_float_gpu = new float[Arows * Bcols];

    // Allocate and initialize host memory for uint8_t matrices
    uint8_t *A_uint8 = new uint8_t[Arows * Acols];
    uint8_t *B_uint8 = new uint8_t[Acols * Bcols];
    uint8_t *C_uint8_cpu = new uint8_t[Arows * Bcols];
    uint8_t *C_uint8_gpu = new uint8_t[Arows * Bcols];

    // Initialize matrices with random data
    srand(time(NULL)); // Seed for random number generation
    for (int i = 0; i < Arows * Acols; ++i)
    {
        A_float[i] = static_cast<float>(rand()) / RAND_MAX;
        A_uint8[i] = static_cast<uint8_t>(rand() % 256);
    }
    for (int i = 0; i < Acols * Bcols; ++i)
    {
        B_float[i] = static_cast<float>(rand()) / RAND_MAX;
        B_uint8[i] = static_cast<uint8_t>(rand() % 256);
    }

    // Perform matrix multiplication on the CPU for float
    auto start = std::chrono::high_resolution_clock::now();
    matMulCPU(A_float, B_float, C_float_cpu, Arows, Acols, Bcols);
    auto finish = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "CPU Time for float: " << elapsed.count() << " s\n";

    // Perform matrix multiplication on the CPU for uint8_t
    start = std::chrono::high_resolution_clock::now();
    matMulCPU(A_uint8, B_uint8, C_uint8_cpu, Arows, Acols, Bcols);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "CPU Time for uint8_t: " << elapsed.count() << " s\n";

    // Allocate device memory
    float *d_A_float, *d_B_float, *d_C_float;
    uint8_t *d_A_uint8, *d_B_uint8, *d_C_uint8;
    cudaMalloc((void **)&d_A_float, Arows * Acols * sizeof(float));
    cudaMalloc((void **)&d_B_float, Acols * Bcols * sizeof(float));
    cudaMalloc((void **)&d_C_float, Arows * Bcols * sizeof(float));
    cudaMalloc((void **)&d_A_uint8, Arows * Acols * sizeof(uint8_t));
    cudaMalloc((void **)&d_B_uint8, Acols * Bcols * sizeof(uint8_t));
    cudaMalloc((void **)&d_C_uint8, Arows * Bcols * sizeof(uint8_t));

    // Copy data from host to device
    cudaMemcpy(d_A_float, A_float, Arows * Acols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_float, B_float, Acols * Bcols * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_A_uint8, A_uint8, Arows * Acols * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B_uint8, B_uint8, Acols * Bcols * sizeof(uint8_t), cudaMemcpyHostToDevice);

    // Perform matrix multiplication on the GPU for float
    start = std::chrono::high_resolution_clock::now();
    matMulGPUFloat(d_A_float, d_B_float, d_C_float, Arows, Acols, Bcols);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "GPU Time for float: " << elapsed.count() << " s\n";

    // Perform matrix multiplication on the GPU for uint8_t
    start = std::chrono::high_resolution_clock::now();
    matMulGPUUint8(d_A_uint8, d_B_uint8, d_C_uint8, Arows, Acols, Bcols);
    finish = std::chrono::high_resolution_clock::now();
    elapsed = finish - start;
    std::cout << "GPU Time for uint8_t: " << elapsed.count() << " s\n";

    // Copy result back to host
    cudaMemcpy(C_float_gpu, d_C_float, Arows * Bcols * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(C_uint8_gpu, d_C_uint8, Arows * Bcols * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    // Clean up
    delete[] A_float;
    delete[] B_float;
    delete[] C_float_cpu;
    delete[] C_float_gpu;
    delete[] A_uint8;
    delete[] B_uint8;
    delete[] C_uint8_cpu;
    delete[] C_uint8_gpu;
    cudaFree(d_A_float);
    cudaFree(d_B_float);
    cudaFree(d_C_float);
    cudaFree(d_A_uint8);
    cudaFree(d_B_uint8);
    cudaFree(d_C_uint8);

    return 0;
}
