import torch
from torch.optim import Optimizer
import math

class ScaledAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, bias_correction=True, adam_w_mode=True,
                 gamma=0.99, amsgrad=False, set_grad_none=True):
        if amsgrad:
            raise RuntimeError('ScaledAdam does not support AMSGrad.')
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        bias_correction=bias_correction,
                        adam_w_mode=adam_w_mode,
                        gamma=gamma)
        super().__init__(params, defaults)
        self.set_grad_none = set_grad_none
        
        # Глобальные параметры масштабирования
        self.sigma_g_sq = 1.0
        self.gamma = gamma

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super().zero_grad()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Собираем градиенты для глобальной оценки дисперсии
        all_grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    all_grads.append(p.grad.detach().flatten())
        
        if len(all_grads) > 0:
            all_grads = torch.cat(all_grads)
            current_var = torch.var(all_grads, unbiased=False).item()
            self.sigma_g_sq = self.gamma * self.sigma_g_sq + (1 - self.gamma) * current_var

        sigma_g = max(math.sqrt(self.sigma_g_sq), 1e-8)
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            
            # Вычисляем коэффициенты масштабирования
            km = math.sqrt(1 - beta1**2) / ((1 - beta1) * sigma_g)
            kv = math.sqrt(1 - beta2**2) / ((1 - beta2) * math.sqrt(2) * sigma_g**2)

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ScaledAdam не поддерживает разреженные градиенты')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']

                # Масштабирование моментов перед обновлением
                exp_avg.mul_(beta1).add_(grad * km, alpha=(1 - beta1))
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=kv*(1 - beta2))

                # print("[DEBUG] exp_avg.std: {}, exp_avg_sq.std: {}".format(exp_avg.std().item(), exp_avg_sq.std().item()))

                # Weight decay (AdamW)
                if group['weight_decay'] != 0 and group['adam_w_mode']:
                    p.data.mul_(1 - lr * group['weight_decay'])

                # Коррекция смещения
                if group['bias_correction']:
                    bias_correction1 = 1 - beta1**t
                    bias_correction2 = 1 - beta2**t
                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    step_size = lr
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                # Обновление параметров
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss