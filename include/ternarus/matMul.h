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
void matMulCPU(const float *A, const float *B, float *C, int Arows, int Acols, int Bcols);

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
void matMulCPU(const uint8_t *A, const uint8_t *B, uint8_t *C, int Arows, int Acols, int Bcols);

/**
 * @brief Multiplies two matrices A and B and stores the result in C on the GPU.
 *
 * @param A Input matrix A (already on the GPU)
 * @param B Input matrix B (already on the GPU)
 * @param C Output matrix C (already on the GPU), where C[i,j] = dot(A[i], B[j])
 * @param Arows Number of rows in A
 * @param Acols Number of columns in A (and rows in B)
 * @param Bcols Number of columns in B
 */
void matMulGPUFloat(const float *A, const float *B, float *C, int Arows, int Acols, int Bcols);

/**
 * @brief Multiplies two matrices A and B and stores the result in C on the GPU.
 *
 * @param A Input matrix A (already on the GPU)
 * @param B Input matrix B (already on the GPU)
 * @param C Output matrix C (already on the GPU), where C[i,j] = dot(A[i], B[j])
 * @param Arows Number of rows in A
 * @param Acols Number of columns in A (and rows in B)
 * @param Bcols Number of columns in B
 */
void matMulGPUUint8(const uint8_t *A, const uint8_t *B, uint8_t *C, int Arows, int Acols, int Bcols);