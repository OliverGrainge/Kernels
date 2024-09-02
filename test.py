import torch
import Ternarus

# CPU usage
A = torch.randn(5)
B = torch.randn(5)
C_cpu = Ternarus.vector_add_cpu(A, B)
print("CPU Result:", C_cpu)

# GPU usage
if torch.cuda.is_available():
    A_gpu = A.cuda()
    B_gpu = B.cuda()
    C_gpu = Ternarus.vector_add_gpu(A_gpu, B_gpu)
    print("GPU Result:", C_gpu)
