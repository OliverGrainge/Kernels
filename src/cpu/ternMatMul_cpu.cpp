// ternMatMul_cpu.cpp
#include <cstdint>
#include "ternarus/ternMatMul.h"

void ternMatMulCPU(const int8_t *A, const float *B, float *C, int Arows, int Acols, int Bcols)
{
    for (int i = 0; i < Arows; i++)
    {
        for (int j = 0; j < Bcols; j++)
        {
            float sum = 0.0f;
            for (int k = 0; k < Acols; k++)
            {
                if (A[i * Acols + k] == 1)
                {
                    sum += B[k * Bcols + j];
                }
                else if (A[i * Acols + k] == -1)
                {
                    sum -= B[k * Bcols + j];
                }
                // If A[i * Acols + k] == 0, do nothing (skip)
            }
            C[i * Bcols + j] = sum;
        }
    }
}
