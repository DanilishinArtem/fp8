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
                    beta = 0.9
                    bias_correction = 1 - beta ** state['step']
                    alpha = (1 - beta) / bias_correction
                    delta_grad = grad - state['exp_avg']
                    d_p = p.data - state['p_prev']
                    denom = d_p.norm(p=4).add(group['eps'])
                    d_p.div_(denom)
                    v_sq = d_p.mul(d_p)
                    delta = delta_grad.div_(denom).mul_(d_p).sum().mul(-alpha) - state['hessian'].mul(v_sq).sum()
                    state['hessian'].addcmul_(v_sq, delta)
                    state['p_prev'].copy_(p.data)
                    state['g_prev'].copy_(grad)
                

                current_hess = state['hessian'].max().item()
                state['cummulative'] += current_hess
                self.writer.add_scalar("hess.mean_{}".format(self.layer), current_hess, self.coutner)
                self.writer.add_scalar("cummulative_hess_{}".format(self.layer), state['cummulative'], self.coutner)

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

                e, m = 4, 3
                # if self.coutner > 1000 and self.coutner < 1200:
                state['exp_avg'] = self.tensor_to_fp8(state['exp_avg'], exponent_bits=e, mantissa_bits=m)
                state['exp_avg_sq'] = self.tensor_to_fp8(state['exp_avg_sq'], exponent_bits=5, mantissa_bits=10)

        return loss
    

    # https://www.math.uwaterloo.ca/~hwolkowi/henry/reports/cauchy.pdf
    # https://arxiv.org/pdf/2009.13586