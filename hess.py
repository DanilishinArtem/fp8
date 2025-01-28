import torch
from torch.optim import Optimizer

class ScaledAdam(Optimizer):
    def __init__(self, params, writer, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, bias_correction=True, adam_w_mode=True,
                 amsgrad=False, set_grad_none=True, hessian_beta=0.99):
        self.writer = writer
        self.counter = 0
        if amsgrad:
            raise RuntimeError('ScaledAdam does not support the AMSGrad variant.')
        
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        bias_correction=bias_correction,
                        adam_w_mode=adam_w_mode,
                        hessian_beta=hessian_beta)
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
        self.counter += 1
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group['betas']
            beta1, beta2 = 0.9, 0.9972337482710926  # Фиксированные значения из оригинального кода
            hessian_beta = group['hessian_beta']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('ScaledAdam does not support sparse gradients')

                state = self.state[p]

                # Инициализация состояния
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['p_prev'] = p.data.clone().detach()
                    state['g_prev'] = grad.clone().detach()
                    state['hessian'] = torch.zeros_like(p.data)
                else:
                    # Вычисление разностей параметров и градиентов
                    s = p.data - state['p_prev']
                    y = grad - state['g_prev']
                    # Оценка диагонали гессиана
                    h_estimate = y / (s + group['eps'])
                    # Обновление гессиана через EMA
                    state['hessian'] = hessian_beta * state['hessian'] + (1 - hessian_beta) * h_estimate
                    # Сохранение текущих значений для следующего шага
                    state['p_prev'].copy_(p.data)
                    state['g_prev'].copy_(grad)

                # Обновление первого момента
                state['exp_avg'].mul_(beta1).add_(grad, alpha=1 - beta1)

                # Подготовка знаменателя с гессианом
                hessian = state['hessian']
                denom = hessian.abs().sqrt().add_(group['eps'])

                # Коррекция смещения
                state['step'] += 1
                t = state['step']
                if group['bias_correction']:
                    bias_correction1 = 1 - beta1 ** t
                    step_size = group['lr'] / bias_correction1
                else:
                    step_size = group['lr']

                # Weight decay (AdamW)
                if group['weight_decay'] != 0 and group['adam_w_mode']:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])

                # Обновление параметров
                p.data.addcdiv_(state['exp_avg'], denom, value=-step_size)

                # Логирование
                self.writer.add_scalar("exp_avg.std", state['exp_avg'].std().item(), self.counter)
                self.writer.add_scalar("hessian.std", hessian.std().item(), self.counter)

        return loss