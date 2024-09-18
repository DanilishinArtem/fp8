import torch
from torch.autograd import Function
from fp8.FP8 import FP8


class FP8GradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Преобразуем тензор в FP8
        fp8_tensor = torch.zeros_like(input, dtype=torch.float16)
        fp8_tensor.copy_(input.to(dtype=torch.float16))
        return fp8_tensor

    @staticmethod
    def backward(ctx, grad_output):
        # Преобразуем градиенты из FP8 обратно в FP32
        return grad_output.to(dtype=torch.float32)

def to_fp8(tensor):
    return FP8GradientFunction.apply(tensor)

def from_fp8(tensor):
    return tensor.to(dtype=torch.float32)


x = torch.rand(1, 10, dtype=torch.float32) * 10000
y = x.clone()
z = x.clone()
print('float32 = {}'.format(x))
y = torch.tensor(y, dtype=torch.float16)
print('float16 = {}'.format(y))
z = to_fp8(z)
print('fp8 = {}'.format(z))