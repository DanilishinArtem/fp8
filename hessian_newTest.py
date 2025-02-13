import numpy as np
import torch
from torch.optim.optimizer import Optimizer


class ScaledAdam(Optimizer):
    def __init__(self, params, writer, lr, beta=0.9, eps=1e-8, rebound='constant', warmup=500, init_lr=None, weight_decay=0, weight_decay_type=None):
        self.layer = None
        self.writer = writer
        self.counter = 0

        if not 0.0 < lr:
            raise ValueError("Invalid learning rate value: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta))
        if rebound not in ['constant', 'belief']:
            raise ValueError("Invalid recitifed bound: {}".format(rebound))
        if not 0.0 <= warmup:
            raise ValueError("Invalid warmup updates: {}".format(warmup))
        if init_lr is None:
            init_lr = lr / 1000
        if not 0.0 <= init_lr <= lr:
            raise ValueError("Invalid initial learning rate: {}".format(init_lr))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if weight_decay_type is None:
            weight_decay_type = 'L2' if rebound == 'constant' else 'decoupled'
        if weight_decay_type not in ['L2', 'decoupled', 'stable']:
            raise ValueError("Invalid weight decay type: {}".format(weight_decay_type))

        defaults = dict(lr=lr, beta=beta, eps=eps, rebound=rebound,
                        warmup=warmup, init_lr=init_lr, base_lr=lr,
                        weight_decay=weight_decay, weight_decay_type=weight_decay_type)
        super(ScaledAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(ScaledAdam, self).__setstate__(state)

    def tensor_to_fp8(self, tensor, exponent_bits=4, mantissa_bits=3):
        max_exponent = 2 ** (exponent_bits - 1) - 1
        min_exponent = -max_exponent + 1
        max_mantissa = 2 ** mantissa_bits - 1
        scale = 2.0 ** min_exponent
        tensor_scaled = tensor / scale
        tensor_quantized = torch.round(tensor_scaled * max_mantissa) / max_mantissa
        tensor_fp8 = tensor_quantized * scale
        return tensor_fp8

    def compute_h_metrics(self, H, prev_H=None, eps=1e-6):
        metrics = {}
        H_flat = H.flatten()
        # H-Sign
        metrics['h_sign'] = torch.sign(H_flat).float().mean()
        # Positive-H-Ratio
        metrics['positive_ratio'] = (H_flat > 0).float().mean()
        # Curvature-SNR
        metrics['curvature_snr'] = torch.abs(H_flat.mean()) / (H_flat.std() + eps)
        # H-Condition (approximated)
        metrics['h_condition'] = H_flat.abs().max() / (H_flat.abs().min() + eps)
        # H-Energy
        metrics['h_energy'] = torch.norm(H, p='fro') / (np.sqrt(H.numel()) + eps)
        # H-Drift (if previous Hessian is available)
        if prev_H is not None:
            metrics['h_drift'] = torch.norm(H - prev_H, p=2)
        
        return metrics


    @torch.no_grad()
    def step(self, closure=None):
        self.layer = 0
        self.counter += 1
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)


                state['step'] += 1

                # Perform optimization step
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Atom does not support sparse gradients.')


                beta, eps = group['beta'], group['eps']
                adam_beta1, adam_beta2 = 0.9, 0.99
                bias_correction1 = 1 - adam_beta1 ** state['step']
                bias_correction2 = 1 - adam_beta2 ** state['step']
                step_size = group['lr'] / bias_correction1

                # Корректное направление обновления (Adam)
                prev_grad = state['exp_avg'].clone()
                state['exp_avg'].mul_(adam_beta1).add_(grad, alpha=1-adam_beta1)
                state['exp_avg_sq'].mul_(adam_beta2).addcmul_(grad, grad, value=1-adam_beta2)

                # Quantization ----------------------------------------------------------------------------------------
                e, m = 5, 2
                state['exp_avg'] = self.tensor_to_fp8(state['exp_avg'], exponent_bits=e, mantissa_bits=m)
                # state['exp_avg_sq'] = self.tensor_to_fp8(state['exp_avg_sq'], exponent_bits=5, mantissa_bits=10)
                # Quantization ----------------------------------------------------------------------------------------

                delta_grad = grad - prev_grad

                denom = (state['exp_avg_sq'].sqrt() / (bias_correction2**0.5)).add_(group['eps'])
                d_p = -step_size * state['exp_avg'] / denom

                p.data.add_(d_p)

                denom = d_p.norm(p=4).add(eps)
                d_p.div_(denom)
                v_sq = d_p.mul(d_p)
                bias_correction = 1 - beta ** state['step']
                alpha = (1 - beta) / bias_correction
                delta = delta_grad.div_(denom).mul_(d_p).sum().mul(-alpha)
                current_hess = v_sq * delta
                metrics = self.compute_h_metrics(current_hess, eps=group['eps'])

                self.layer += 1
                # self.writer.add_scalar("hessNorm_{}".format(self.layer), current_hess.norm().item(), self.counter)
                self.writer.add_scalar("h_sign_{}".format(self.layer), metrics['h_sign'].item(), self.counter)
                self.writer.add_scalar("positive_ratio_{}".format(self.layer), metrics['positive_ratio'].item(), self.counter)
                # self.writer.add_scalar("curvature_snr_{}".format(self.layer), metrics['curvature_snr'].item(), self.counter)
                self.writer.add_scalar("h_condition_{}".format(self.layer), metrics['h_condition'].item(), self.counter)
                self.writer.add_scalar("h_energy_{}".format(self.layer), metrics['h_energy'].item(), self.counter)

        return loss
    

# https://arxiv.org/pdf/2412.05270
# https://arxiv.org/pdf/2009.13586