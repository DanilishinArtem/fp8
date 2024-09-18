import torch
from torch.autograd import Function
from fp8.FP8 import FP8


class FP8E4M3Tensor(torch.autograd.Function):
    EXPONENT_BITS = 4
    MANTISSA_BITS = 3
    EXPONENT_BIAS = 7

    SIGN_BIT_MASK = 0x80  # 1000 0000 (знак)
    EXPONENT_MASK = 0x78  # 0111 1000 (экспонента)
    MANTISSA_MASK = 0x07  # 0000 0111 (мантисса)

    MAX_EXPONENT = 0xF  # 1111 (макс значение экспоненты)
    MIN_EXPONENT = 0x1  # 0001 (мин нормальное значение)

    @staticmethod
    def forward(ctx, input):
        # Преобразуем входной FP32 тензор в FP8
        output = torch.empty_like(input, dtype=torch.uint8)

        # Для каждого элемента преобразуем в FP8
        for i in range(input.numel()):
            output.view(-1)[i] = FP8E4M3Tensor.fp32_to_fp8(input.view(-1)[i])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Преобразуем градиенты из FP8 обратно в FP32
        grad_input = torch.empty_like(grad_output, dtype=torch.float32)
        for i in range(grad_output.numel()):
            grad_input.view(-1)[i] = FP8E4M3Tensor.fp8_to_fp32(grad_output.view(-1)[i])

        return grad_input

    @staticmethod
    def fp32_to_fp8(value):
        """Преобразование FP32 в FP8 (E4M3) с правилами для нормальных и субнормальных чисел"""
        if value == 0.0:
            return torch.tensor(0, dtype=torch.uint8)  # Представление нуля

        # Извлекаем знак
        sign = 0
        if value < 0:
            sign = 1
            value = -value

        # Экспонента и нормализация значения
        exponent_raw = torch.floor(torch.log2(value)).item()
        mantissa_raw = (value / (2 ** exponent_raw)) - 1.0

        exponent = int(exponent_raw + FP8E4M3Tensor.EXPONENT_BIAS)
        mantissa = int(mantissa_raw * (2 ** FP8E4M3Tensor.MANTISSA_BITS))

        # Обработка нормальных значений
        if exponent >= FP8E4M3Tensor.MIN_EXPONENT:
            # Нормальные числа
            if exponent > FP8E4M3Tensor.MAX_EXPONENT:
                exponent = FP8E4M3Tensor.MAX_EXPONENT
                mantissa = 0
        else:
            # Субнормальные числа
            exponent = 0
            mantissa = int(value * (2 ** (6 + FP8E4M3Tensor.MANTISSA_BITS)))

        # Формируем финальный FP8 формат
        fp8_value = (sign << 7) | (exponent << FP8E4M3Tensor.MANTISSA_BITS) | mantissa
        return torch.tensor(fp8_value, dtype=torch.uint8)

    @staticmethod
    def fp8_to_fp32(fp8_value):
        """Преобразуем FP8 обратно в FP32"""
        # Извлекаем знак, экспоненту и мантиссу
        sign = (fp8_value >> 7) & 0x01
        exponent = (fp8_value >> 3) & 0x0F
        mantissa = fp8_value & 0x07

        if exponent == 0:
            # Субнормальные числа
            value = mantissa / (2 ** (6 + FP8E4M3Tensor.MANTISSA_BITS))
        else:
            # Нормализованные числа
            exp = exponent - FP8E4M3Tensor.EXPONENT_BIAS
            value = (1 + mantissa / (2 ** FP8E4M3Tensor.MANTISSA_BITS)) * (2 ** exp)

        if sign:
            value = -value

        return value

# Вспомогательные функции для преобразования тензоров
def to_fp8(tensor):
    """Преобразуем тензор FP32 в FP8"""
    return FP8E4M3Tensor.apply(tensor)

def from_fp8(tensor):
    """Преобразуем тензор FP8 обратно в FP32"""
    return tensor.to(dtype=torch.float32)


x = torch.rand(1, 10, dtype=torch.float32) * 10
y = x.clone()
z = x.clone()
print('float32 = {}'.format(x))
y = torch.tensor(y, dtype=torch.float16)
print('float16 = {}'.format(y))
z = to_fp8(z)
print('fp8 = {}'.format(z))