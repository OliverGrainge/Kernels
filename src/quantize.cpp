// src/quantize.cpp
#include "quantize.h"

torch::Tensor quantize_tensor(torch::Tensor tensor, float scale) {
    // Example quantization: divide the tensor by the scale and round
    return (tensor / scale).round();
}