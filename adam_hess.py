import torch
from torch.optim import Optimizer

class ScaledAdam(Optimizer):
    def __init__(self, params, writer, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, bias_correction=True, adam_w_mode=True,
                 amsgrad=False, set_grad_none=True):
        self.layer = None
        self.writer = writer
        self.coutner = 0
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
        self.coutner += 1
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
                    state['p_prev'] = p.data.clone().detach()
                    state['g_prev'] = grad.clone().detach()
                    state['hessian'] = torch.zeros_like(p.data)
                    state['cummulative'] = 0
                else:
                    delta_p = p.data - state['p_prev']
                    delta_g = grad - state['g_prev']
                    # d = -delta_p / group['lr']
                    state['hessian'] = state['hessian'] + (((delta_p * delta_g).sum() - (delta_p * state['hessian'] * delta_p).sum()) / delta_p.norm(p=4)) * delta_p.pow(2)
                    state['p_prev'] = p.data.clone().detach()
                    state['g_prev'] = grad.clone().detach()
                    # print("Number of elements in hessian: {}".format(state['hessian'].numel()))

                    ind = (state['hessian'] * delta_p.pow(2)).mean() * group['lr'] * pow(beta2, 1/2) / beta1 / grad.norm(p=1) / 2
                    current_hess = state['hessian'].max().item()
                    state['cummulative'] += current_hess

                    self.writer.add_scalar("hessianMax_{}".format(self.layer), current_hess, self.coutner)
                    self.writer.add_scalar("cummulative_hess_{}".format(self.layer), state['cummulative'], self.coutner)
                    self.writer.add_scalar("ind_{}".format(self.layer), ind, self.coutner)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # Применяем weight decay (AdamW vs Adam)
                if group['weight_decay'] != 0:
                    if group['adam_w_mode']:
                        p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    else:
                        grad.add_(p.data, alpha=group['weight_decay'])

                # Обновляем моменты
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Коррекция смещения
                if group['bias_correction']:
                    bias_correction1 = 1 - beta1 ** t
                    bias_correction2 = 1 - beta2 ** t
                    step_size = group['lr'] / bias_correction1
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(group['eps'])
                else:
                    step_size = group['lr']
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Обновление параметров
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss
    

    # https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/cauchy.pdf
    # https://arxiv.org/pdf/2009.13586