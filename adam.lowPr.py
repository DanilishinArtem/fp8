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
                beta1, beta2 = 0.9, 0.9972337482710926
                self.layer += 1
                # Инициализация состояния
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['sigma_g_sq'] = group['eps']
                    state['gamma'] = 0.999

                
                # Part of gradient correction ..................................................................................................
                # state['sigma_g_sq'] = state['gamma'] * state['sigma_g_sq'] + (1 - state['gamma']) * p.grad.data.var().item()
                state['sigma_g_sq'] = state['gamma'] * state['sigma_g_sq'] + (1 - state['gamma']) * (p.grad.data * p.grad.data).mean().item()
                
                # Масштабирование первого момента
                km_numerator = pow(1 - beta1, 2) * state['sigma_g_sq']
                km_denominator = 1 - beta1**2
                # km = km_numerator / (km_denominator + 1e-16)
                km = km_numerator / (km_denominator)
                # Масштабирование второго момента (предполагаем нормальность градиентов)
                kv_numerator = pow(1 - beta2, 2) * state['sigma_g_sq'] * state['sigma_g_sq'] * 2
                kv_denominator = 1 - beta2**2
                # kv = kv_numerator / (kv_denominator + 1e-16)
                kv = kv_numerator / (kv_denominator)
                # Part of gradient correction ..................................................................................................

                state['step'] += 1
                t = state['step']

                # Применяем weight decay (AdamW vs Adam)
                if group['weight_decay'] != 0:
                    if group['adam_w_mode']:
                        p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    else:
                        grad.add_(p.data, alpha=group['weight_decay'])

                
                # # Обновляем моменты
                # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state['exp_avg'].mul_(beta1).add_(grad * pow(1 / km, 1 / 2), alpha=1 - beta1)
                state['exp_avg_sq'].mul_(beta2).addcmul_(grad * pow(1 / kv, 1 / 2), grad, value=1 - beta2)
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
                # N = 1e6
                N = 1e20
                p.data.addcdiv_(state['exp_avg'] / pow(N, 1/2), denom, value=-step_size)

                # # Part of casting to FP8
                # e, m = 5, 2
                # state['exp_avg'] = self.tensor_to_fp8(state['exp_avg'], exponent_bits=e, mantissa_bits=m)
                # state['exp_avg_sq'] = self.tensor_to_fp8(state['exp_avg_sq'], exponent_bits=e, mantissa_bits=m)

                # Part of castirng to FP4
                state['exp_avg'] = self.tensor_to_fp8(state['exp_avg'], exponent_bits=2, mantissa_bits=1)
                state['exp_avg_sq'] = self.tensor_to_fp8(state['exp_avg_sq'], exponent_bits=2, mantissa_bits=1)

                # self.writer.add_histogram("exp_avg_layer_{}".format(self.layer), state['exp_avg'], self.counter)
                # self.writer.add_histogram("exp_avg_sq_layer_{}".format(self.layer), state['exp_avg_sq'], self.counter)


                ind = state['exp_avg']
                # ind = state['exp_avg_sq']
                # ind = state['exp_avg'] * denom / step_size
                
                self.writer.add_scalar("ind_min[{}]".format(self.layer), ind.min().item(), self.counter)
                self.writer.add_scalar("ind_max[{}]".format(self.layer), ind.max().item(), self.counter)
                self.writer.add_scalar("ind_mean[{}]".format(self.layer), ind.mean().item(), self.counter)
        return loss
    

    # https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/cauchy.pdf
    # https://arxiv.org/pdf/2009.13586