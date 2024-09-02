// bindings.cpp
#include <torch/extension.h>
#include "ternarus/vectorAdd.h"

torch::Tensor vector_add_cpu(torch::Tensor A, torch::Tensor B)
{
    auto N = A.size(0);
    auto C = torch::empty_like(A);

    vectorAddCPU(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C;
}

torch::Tensor vector_add_gpu(torch::Tensor A, torch::Tensor B)
{
    // Ensure inputs are on GPU
    TORCH_CHECK(A.is_cuda(), "Tensor A must be on GPU");
    TORCH_CHECK(B.is_cuda(), "Tensor B must be on GPU");

    auto N = A.size(0);
    auto C = torch::empty_like(A); // Allocate C on GPU

    vectorAddGPU(A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), N);

    return C; // Return the result, which is still on the GPU
}

PYBIND11_MODULE(Ternarus, m)
{
    m.def("vector_add_cpu", &vector_add_cpu, "Vector addition on CPU");
    m.def("vector_add_gpu", &vector_add_gpu, "Vector addition on GPU");
}
