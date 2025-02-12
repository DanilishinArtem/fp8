import torch
from torch.optim import Optimizer

class ScaledAdam(Optimizer):
    def __init__(self, params, writer, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, bias_correction=True, adam_w_mode=True,
                 amsgrad=False, set_grad_none=True):
        self.layer = None
        self.writer = writer
        self.counter = 0
        if amsgrad:
            raise RuntimeError('AdamNoApex does not support the AMSGrad variant.')
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        bias_correction=bias_correction,
                        adam_w_mode=adam_w_mode) 
        super().__init__(params, defaults)
        self.set_grad_none = set_grad_none

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super().zero_grad()

    def tensor_to_fp8(self, tensor, exponent_bits=4, mantissa_bits=3):
        max_exponent = 2 ** (exponent_bits - 1) - 1
        min_exponent = -max_exponent + 1
        max_mantissa = 2 ** mantissa_bits - 1
        scale = 2.0 ** min_exponent
        tensor_scaled = tensor / scale
        tensor_quantized = torch.round(tensor_scaled * max_mantissa) / max_mantissa
        tensor_fp8 = tensor_quantized * scale
        return tensor_fp8

    def step(self, closure=None):
        self.layer = 0
        self.counter += 1
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('AdamNoApex does not support sparse gradients')

                state = self.state[p]
                beta1, beta2 = group['betas']
                self.layer += 1
                # Инициализация состояния
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['scale'] = None

                state['step'] += 1
                t = state['step']

                # # Применяем weight decay (AdamW vs Adam)
                # if group['weight_decay'] != 0:
                #     if group['adam_w_mode']:
                #         p.data.mul_(1 - group['lr'] * group['weight_decay'])
                #     else:
                #         grad.add_(p.data, alpha=group['weight_decay'])

                
                # # Обновляем моменты
                # if state['scale'] == None:
                #     state['scale'] = grad.std()
                # grad /= state['scale']
                # beta_temp = 1 / (pow(beta1, 2) + pow(1 - beta1, 2))
                # state['exp_avg_sq'] = -(2 * beta1 * (1 - beta1) * beta_temp) * state['exp_avg'] * grad
                # state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                # state['exp_avg_sq'] = state['exp_avg_sq'] + state['exp_avg'].pow(2) * beta_temp

                fact = 5.5 / (grad.abs().max().item())
                # fact = 1.0 / grad.std().item()
                self.writer.add_scalar("fact_{}".format(self.layer), fact, self.counter)
                # grad = grad / fact

                # Part of casting to FP8
                # e, m = 5, 2
                # e, m = 2, 1
                # state['exp_avg'] = self.tensor_to_fp8(state['exp_avg'], exponent_bits=e, mantissa_bits=m)
                # state['exp_avg_sq'] = self.tensor_to_fp8(state['exp_avg_sq'], exponent_bits=e, mantissa_bits=m)

                state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # self.writer.add_scalar("exp_avg_sq_layer_{}".format(self.layer), state['exp_avg_sq'].std().item(), self.counter)


                # Коррекция смещения
                if group['bias_correction']:
                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t
                    step_size = group['lr'] / bias_correction1
                    denom = (state['exp_avg_sq'].sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                else:
                    step_size = group['lr']
                    denom = state['exp_avg_sq'].sqrt().add_(group['eps'])

                # Обновление параметров
                p.data.addcdiv_(state['exp_avg'], denom, value=-step_size)


                # self.writer.add_histogram("exp_avg_layer_{}".format(self.layer), state['exp_avg'], self.counter)
                # self.writer.add_histogram("exp_avg_sq_layer_{}".format(self.layer), state['exp_avg_sq'], self.counter)


                # ind = state['exp_avg']
                # ind = state['exp_avg_sq']
                # ind = state['exp_avg'] * denom / step_size
                
                # self.writer.add_scalar("abs_min[{}]".format(self.layer), ind.abs().min().item(), self.counter)
                # self.writer.add_scalar("abs_max[{}]".format(self.layer), ind.abs().max().item(), self.counter)
        return loss
    

    # https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/cauchy.pdf
    # https://arxiv.org/pdf/2009.13586