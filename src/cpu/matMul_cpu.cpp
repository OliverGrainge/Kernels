// matMul_cpu.cpp
#include "ternarus/matMul.h"
#include <cstdint>

void matMulCPU(const float *A, const float *B, float *C, int Arows, int Acols, int Bcols)
{
    for (int i = 0; i < Arows; i++)
    {
        for (int j = 0; j < Bcols; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < Acols; k++)
            {
                sum += A[i * Acols + k] * B[k * Bcols + j];
            }
            C[i * Bcols + j] = sum;
        }
    }
}

void matMulCPU(const uint8_t *A, const uint8_t *B, uint8_t *C, int Arows, int Acols, int Bcols)
{
    for (int i = 0; i < Arows; i++)
    {
        for (int j = 0; j < Bcols; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < Acols; k++)
            {
                sum += A[i * Acols + k] * B[k * Bcols + j];
            }
            C[i * Bcols + j] = sum;
        }
    }
}
