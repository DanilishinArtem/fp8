import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import scipy.stats as stats


class GradientCollector:
    def __init__(self) -> None:
        self.k = 10
        self.alpha = 0.0
        self.alpha_1 = 1.0 - self.alpha
        self.beta = 0.99
        self.beta_1 = 1.0 - self.beta
        self.mean = None
        self.var = torch.tensor(0.0)
        self.timer = 0
        self.first = True

    def collectGradients(self, model: torch.nn.Module, writer: SummaryWriter, total_counter: int):
        name = 0
        self.timer += 1
        for module in model.parameters():
            if module.requires_grad:
                name += 1
                grads = torch.abs(module.grad.detach().clone())
                mask = grads > 0
                temp = torch.aminmax(grads[mask])
                MinMax = torch.log10(temp.max / temp.min)
                if self.mean is None:
                    self.mean = MinMax
                    return
                
                bound = self.mean + self.k * torch.sqrt(self.var)
                if temp.max > bound:
                    return
                self.var = self.beta * self.var + (1 - self.beta) * pow(MinMax - self.mean, 2)
                self.mean = torch.log10(MinMax)

                if self.timer > 10:
                    writer.add_scalars('bound' + str(str(name)), {
                        'boundUp': bound.item(),
                        'currectValue': MinMax.item(),
                    }, total_counter)
        self.first = False
