import torch
from mnist_analisys.config import Config
import struct
from torch.utils.tensorboard import SummaryWriter
from scipy.stats import gamma
import numpy as np


hashMap = {}

config = Config()

def bitflip(x, position):
    packed = struct.pack('!f', x)
    bits = struct.unpack('!I', packed)[0]
    exponent_mask = 0x7F800000
    exponent = (bits & exponent_mask) >> 23
    bit_to_flip = 1 << position
    exponent ^= bit_to_flip
    new_bits = (bits & ~exponent_mask) | (exponent << 23)
    new_packed = struct.pack('!I', new_bits)
    new_x = struct.unpack('!f', new_packed)[0]
    return new_x


def fault_function(input: torch.tensor):
    if config.rate == 0:
        random_indices = torch.randint(0, input.numel(), (1,))
    else:
        random_indices = torch.randint(0, input.numel(), (int(input.numel() * config.rate),))
    random_positions = torch.unravel_index(random_indices, input.shape)
    temp = input[random_positions]
    print("[FAULT INJECTION]")
    print(temp)
    for i in range(len(temp)):
        temp[i] = bitflip(temp[i], config.position)
    print("-------------------->")
    print(temp)
    input[random_positions] = temp
    return input

# --------------------------------------------------------------

class Hook:
    def __init__(self, name: str, writer):
        self.name = name
        self.counter = -1
        self.writer = writer

    def hook(self, grad):
        self.counter += 1
        if self.name == config.target and self.counter in config.fault_time:
            print("fault for layer " + self.name + " was injected")
            grad = fault_function(grad)
        checkGrad(grad, self.name, self.counter, self.writer)
        return grad
    

class HookSetter:
    def __init__(self, model: torch.nn.Module, writer):
        self.setter(model, writer)

    def setter(self, model: torch.nn.Module, writer):
        counter = 0
        for module in model.parameters():
            if module.requires_grad:
                counter += 1
                print("registered hook for layer: " + str(counter))
                Hook_layer = Hook(str(counter), writer)
                module.register_hook(Hook_layer.hook)


def checkGrad(grad: torch.Tensor, name: str, time: int, writer: SummaryWriter):
    pass