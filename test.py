import torch
import mymodule

tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
scale = 0.5

quantized_tensor = mymodule.quantize_tensor(tensor, scale)
print(quantized_tensor)  # The tensor elements will be quantized based on the scale
