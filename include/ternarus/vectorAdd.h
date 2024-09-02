// vectorAdd.h
#pragma once

/**
 * @brief Adds two vectors A and B and stores the result in C on the CPU.
 *
 * @param A Input vector A
 * @param B Input vector B
 * @param C Output vector C, where C[i] = A[i] + B[i]
 * @param N Number of elements in vectors A, B, and C
 */
void vectorAddCPU(const float *A, const float *B, float *C, int N);

/**
 * @brief Adds two vectors A and B and stores the result in C on the GPU.
 *
 * @param A Input vector A
 * @param B Input vector B
 * @param C Output vector C, where C[i] = A[i] + B[i]
 * @param N Number of elements in vectors A, B, and C
 */
void vectorAddGPU(const float *A, const float *B, float *C, int N);
