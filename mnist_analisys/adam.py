import torch
from torch.optim import Optimizer

class ScaledAdam(Optimizer):
    def __init__(self, params, writer, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, bias_correction=True, adam_w_mode=True,
                 amsgrad=False, set_grad_none=True):
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

    def step(self, closure=None):
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
                beta1, beta2 = 0.9, 0.9972337482710926

                # Инициализация состояния
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    state['p_prev'] = p.data.clone().detach()
                    state['g_prev'] = grad.clone().detach()
                    state['hessian'] = torch.zeros_like(p.data)

                    state['sigma_g_sq'] = group['eps']
                    state['gamma'] = 0.999
                else:
                    hessian_beta=0.99
                    s = p.data - state['p_prev']
                    y = grad - state['g_prev']
                    # Оценка диагонали гессиана
                    h_estimate = y / (s + group['eps'])
                    # Обновление гессиана через EMA
                    state['hessian'] = hessian_beta * state['hessian'] + (1 - hessian_beta) * h_estimate
                    # Сохранение текущих значений для следующего шага
                    state['p_prev'].copy_(p.data)
                    state['g_prev'].copy_(grad)
                
                # Part of gradient correction ..................................................................................................
                state['sigma_g_sq'] = state['gamma'] * state['sigma_g_sq'] + (1 - state['gamma']) * p.grad.data.var().item()
                sigma_g_sq = state['sigma_g_sq']
                sigma_g = pow(sigma_g_sq + 1e-16, 1 / 2)  # Добавляем для стабильности
                
                # Масштабирование первого момента
                km_numerator = pow(1 - beta1, 2) * sigma_g_sq
                km_denominator = 1 - beta1**2
                km = km_numerator / (km_denominator + 1e-16)
                
                # Масштабирование второго момента (предполагаем нормальность градиентов)
                kv_numerator = pow(1 - beta2, 2) * sigma_g_sq * sigma_g_sq * 2
                kv_denominator = 1 - beta2**2
                kv = kv_numerator / (kv_denominator + 1e-16)
                # Part of gradient correction ..................................................................................................

                self.writer.add_scalar("exp_avg.std", state['exp_avg'].std().item(), self.coutner)
                self.writer.add_scalar("exp_avg_sq.std", state['exp_avg_sq'].std().item(), self.coutner)
                self.writer.add_scalar("hess.min", state['hessian'].min().item(), self.coutner)
                self.writer.add_scalar("hess.max", state['hessian'].max().item(), self.coutner)
                self.writer.add_scalar("hess.mean", state['hessian'].mean().item(), self.coutner)

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
                exp_avg.mul_(beta1).add_(grad * pow(1 / km, 1 / 2), alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad * pow(1 / kv, 1 / 2), grad, value=1 - beta2)
                # exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

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