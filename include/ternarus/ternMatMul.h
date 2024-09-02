#pragma once
#include <cstdint>

/**
 * @brief Multiplies two matrices A and B and stores the result in C on the CPU.
 *
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C, where C[i,j] = dot(A[i], B[j])
 * @param Arows Number of rows in A
 * @param Acols Number of columns in A (and rows in B)
 * @param Bcols Number of columns in B
 */
void ternMatMulCPU(const int8_t *A, const float *B, float *C, int Arows, int Acols, int Bcols);

/**
 * @brief Multiplies two matrices A and B and stores the result in C on the GPU.
 *
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C, where C[i,j] = dot(A[i], B[j])
 * @param Arows Number of rows in A
 * @param Acols Number of columns in A (and rows in B)
 * @param Bcols Number of columns in B
 */
void ternMatMulGPU(const int8_t *A, const float *B, float *C, int Arows, int Acols, int Bcols);

/**
 * @brief Multiplies a bitpacked ternary matrix A and matrix B and stores the result in C on the GPU.
 *
 * @param packedA Bitpacked input matrix A
 * @param B Input matrix B
 * @param C Output matrix C, where C[i,j] = dot(A[i], B[j])
 * @param Arows Number of rows in A
 * @param Acols Number of columns in A (and rows in B)
 * @param Bcols Number of columns in B
 */
extern "C" void ternMatMulBitpackedGPU(const uint32_t *packedA, const float *B, float *C, int Arows, int Acols, int Bcols);
