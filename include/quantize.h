// include/quantize.h
#pragma once

#include <torch/torch.h>  // Include PyTorch's header

torch::Tensor quantize_tensor(torch::Tensor tensor, float scale);
