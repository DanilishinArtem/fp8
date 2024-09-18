import torch
# https://github.com/NVIDIA/TransformerEngine/issues/965
# https://github.com/NVIDIA/TransformerEngine/blob/744624d004f4514ffbaa90ac83e214311c86c607/transformer_engine/pytorch/float8_tensor.py#L12

# Constants for FP8 format
SIGN_BIT = 0x80
EXPONENT_MASK = 0x7C
MANTISSA_MASK = 0x03

def float_to_fp8(value):
    """Convert a float to FP8 representation."""
    if value == 0:
        return torch.tensor(0, dtype=torch.uint8)

    # Convert to float32
    fp32_value = torch.tensor(value, dtype=torch.float32)
    
    # Extract sign bit
    sign = 0
    if fp32_value < 0:
        sign = 1
        fp32_value = -fp32_value

    # Extract exponent and mantissa
    exponent_bias = 15
    exponent_raw = torch.floor(torch.log2(fp32_value)).item()
    mantissa_raw = (fp32_value / (2 ** exponent_raw)) - 1.0
    
    exponent = int(exponent_raw + exponent_bias)
    mantissa = int(mantissa_raw * 8)  # 3-bit mantissa
    
    # Handle overflow and underflow
    max_exp = 15
    if exponent >= max_exp:
        exponent = max_exp
        mantissa = 0
    elif exponent <= 0:
        exponent = 0
        mantissa = 0
    
    # Pack into FP8 format
    fp8_value = (sign << 7) | (exponent << 3) | mantissa
    return torch.tensor(fp8_value, dtype=torch.uint8)

def fp8_to_float(fp8_value):
    """Convert FP8 representation back to float."""
    # Extract sign, exponent, and mantissa
    sign = (fp8_value >> 7) & 0x01
    exponent = (fp8_value >> 3) & 0x0F
    mantissa = fp8_value & 0x07
    
    # Handle special cases
    if exponent == 0:
        return 0.0
    if exponent == 0x0F:
        return float('inf')

    # Compute the floating-point value
    exponent_bias = 15
    exp = exponent - exponent_bias
    value = (1 + (mantissa / 8)) * (2 ** exp)
    if sign:
        value = -value

    return value

class FP8:
    def __init__(self, value):
        self.value = float_to_fp8(value)

    def to_float(self):
        return fp8_to_float(self.value)

    def __add__(self, other):
        return FP8(self.to_float() + other.to_float())

    def __sub__(self, other):
        return FP8(self.to_float() - other.to_float())

    def __mul__(self, other):
        return FP8(self.to_float() * other.to_float())

    def __truediv__(self, other):
        return FP8(self.to_float() / other.to_float())

    def __repr__(self):
        return f"FP8({self.to_float()})"


