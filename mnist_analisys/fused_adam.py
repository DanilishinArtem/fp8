import torch
from apex.multi_tensor_apply import multi_tensor_applier

class FusedAdam(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, bias_correction=True,
                 betas=(0.9, 0.999), eps=1e-8, adam_w_mode=True,
                 weight_decay=0., amsgrad=False, set_grad_none=True,
                 capturable=False, master_weights=False):
        print("Initialazing FusedAdam with standartization...")
        if amsgrad:
            raise RuntimeError('FusedAdam does not support the AMSGrad variant.')
        if master_weights and not capturable:
            raise RuntimeError('Master weights is currently only supported with the capturable version.')
        # If the optimizer is capturable then LR should be a tensor (on GPU)
        lr = torch.tensor(lr, dtype=torch.float32) if capturable else lr
        defaults = dict(lr=lr, bias_correction=bias_correction,
                        betas=betas, eps=eps, weight_decay=weight_decay)
        super(FusedAdam, self).__init__(params, defaults)
        self.adam_w_mode = 1 if adam_w_mode else 0
        self.set_grad_none = set_grad_none

        self.capturable = capturable
        self.master_weights = master_weights

        # Create full precision master weights
        self.param_groups_master = []
        for i, pg in enumerate(self.param_groups):
            param_list = pg['params']
            self.param_groups_master.append({
                'params': [
                    p.clone().detach().float() if self.master_weights else None
                    for p in param_list
                ],
            })

        if capturable:
            for idx, group in enumerate(self.param_groups):
                if len(group['params']) == 0:
                    continue
                device = group['params'][0].device
                for item in ['lr']:
                    self.param_groups[idx][item] = group[item].to(device=device)

            self._step_supports_amp_scaling = True

        if multi_tensor_applier.available:
            import amp_C
            # Skip buffer
            self._dummy_overflow_buf = torch.tensor([0], dtype=torch.int, device='cuda')
            self.multi_tensor_adam = amp_C.multi_tensor_adam
            self.multi_tensor_adam_capturable = amp_C.multi_tensor_adam_capturable
            self.multi_tensor_adam_capturable_master = amp_C.multi_tensor_adam_capturable_master
        else:
            raise RuntimeError('apex.optimizers.FusedAdam requires cuda extensions')

    def zero_grad(self):
        if self.set_grad_none:
            for group in self.param_groups:
                for p in group['params']:
                    p.grad = None
        else:
            super(FusedAdam, self).zero_grad()

    def step(self, closure=None, grads=None, output_params=None, scale=None, grad_norms=None, grad_scaler=None):

        if any(p is not None for p in [grads, output_params, scale, grad_norms]):
            raise RuntimeError('FusedAdam has been updated.  Simply initialize it identically to torch.optim.Adam, and call step() with no arguments.')
        loss = None
        if closure is not None:
            loss = closure()

        for group, group_master in zip(self.param_groups, self.param_groups_master):
            if len(group['params']) == 0:
                continue
            device = group['params'][0].device
            bias_correction = 1 if group['bias_correction'] else 0
            beta1, beta2 = group['betas']

            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1 if not self.capturable else (self._dummy_overflow_buf != 1).to(torch.int)
            else:
                group['step'] = 1 if not self.capturable else torch.tensor([1], dtype=torch.int, device=device)

            # create lists for multi-tensor apply
            g_16, p_16, m_16, v_16 = [], [], [], []
            g_bf, p_bf, m_bf, v_bf = [], [], [], []
            g_32, p_32, m_32, v_32 = [], [], [], []
            p_16_master = []
            p_32_master = []

            for p, p_master in zip(group['params'], group_master['params']):
                if p.grad is None:
                    continue
                if p.grad.data.is_sparse:
                    raise RuntimeError('FusedAdam does not support sparse gradients, please consider SparseAdam instead')

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data).float()
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data).float()
                    state['sigma_g_sq'] = 0
                    group['gamma'] = 0.99
                
                group['sigma_g_sq'] += group['gamma'] * group['sigma_g_sq'] + (1 - group['gamma']) * p.grad.data.var().item()
                k_m = pow((1 + beta1) / (1 - beta1), 1 / 2) / pow(group['sigma_g_sq'], 1 / 2)
                k_v = pow((1 + beta2) / (1 - beta2), 1 / 2) / pow(2 * group['sigma_g_sq'], 1 / 2)
                state['exp_avg'].mul_(k_m)
                state['exp_avg_sq'].mul_(k_v)
                print("[DEBUG] exp_avg.std = {}, exp_avg_sq.std = {}".format(state['exp_avg'].std().item(), state['exp_avg_sq'].std().item()))

                if p.dtype == torch.float16:
                    if self.master_weights:
                        p_16_master.append(p_master.data)
                    g_16.append(p.grad.data)
                    p_16.append(p.data)
                    m_16.append(state['exp_avg'])
                    v_16.append(state['exp_avg_sq'])
                elif p.dtype == torch.bfloat16:
                    g_bf.append(p.grad)
                    p_bf.append(p)
                    m_bf.append(state['exp_avg'])
                    v_bf.append(state['exp_avg_sq'])
                elif p.dtype == torch.float32:
                    if self.master_weights:
                        p_32_master.append(p_master.data)
                    g_32.append(p.grad.data)
                    p_32.append(p.data)
                    m_32.append(state['exp_avg'])
                    v_32.append(state['exp_avg_sq'])
                else:
                    raise RuntimeError('FusedAdam only support fp16 and fp32.')

            # If the optimizer is capturable, then if there's a grad scaler it works
            # on the GPU + a different multi_tensor_applier should be called
            if self.capturable:
                # overflow check of gradients
                found_inf = (
                    grad_scaler._check_inf_per_device(self)[device]
                    if grad_scaler is not None else torch.zeros((1,), device=device)
                )
                self._dummy_overflow_buf.copy_(found_inf)

                # get unscale scale factor
                scale, inv_scale = None, None
                if grad_scaler:
                    scale = grad_scaler._get_scale_async()
                    inv_scale = scale.double().reciprocal().float()
                else:
                    scale = torch.ones((1,), device=device)
                    inv_scale = torch.ones((1,), device=device)

                if len(g_16) > 0:
                    multi_tensor_applier(self.multi_tensor_adam_capturable_master if self.master_weights
                            else self.multi_tensor_adam_capturable,
                            self._dummy_overflow_buf,
                            [g_16, p_16, m_16, v_16, p_16_master] if self.master_weights
                            else [g_16, p_16, m_16, v_16],
                            group['lr'],
                            beta1,
                            beta2,
                            group['eps'],
                            group['step'],
                            self.adam_w_mode,
                            bias_correction,
                            group['weight_decay'],
                            inv_scale)

                if len(g_bf) > 0:
                    multi_tensor_applier(
                            self.multi_tensor_adam_capturable,
                            self._dummy_overflow_buf,
                            [g_bf, p_bf, m_bf, v_bf],
                            group['lr'],
                            beta1,
                            beta2,
                            group['eps'],
                            group['step'],
                            self.adam_w_mode,
                            bias_correction,
                            group['weight_decay'],
                            inv_scale)

                if len(g_32) > 0:
                    multi_tensor_applier(self.multi_tensor_adam_capturable_master if self.master_weights
                            else self.multi_tensor_adam_capturable,
                            self._dummy_overflow_buf,
                            [g_32, p_32, m_32, v_32, p_32_master] if self.master_weights
                            else [g_32, p_32, m_32, v_32],
                            group['lr'],
                            beta1,
                            beta2,
                            group['eps'],
                            group['step'],
                            self.adam_w_mode,
                            bias_correction,
                            group['weight_decay'],
                            inv_scale)
            else:
                if len(g_16) > 0:
                    multi_tensor_applier(self.multi_tensor_adam,
                            self._dummy_overflow_buf,
                            [g_16, p_16, m_16, v_16],
                            group['lr'],
                            beta1,
                            beta2,
                            group['eps'],
                            group['step'],
                            self.adam_w_mode,
                            bias_correction,
                            group['weight_decay'])

                if len(g_bf) > 0:
                    multi_tensor_applier(
                            self.multi_tensor_adam,
                            self._dummy_overflow_buf,
                            [g_bf, p_bf, m_bf, v_bf],
                            group['lr'],
                            beta1,
                            beta2,
                            group['eps'],
                            group['step'],
                            self.adam_w_mode,
                            bias_correction,
                            group['weight_decay'])

                if len(g_32) > 0:
                    multi_tensor_applier(self.multi_tensor_adam,
                            self._dummy_overflow_buf,
                            [g_32, p_32, m_32, v_32],
                            group['lr'],
                            beta1,
                            beta2,
                            group['eps'],
                            group['step'],
                            self.adam_w_mode,
                            bias_correction,
                            group['weight_decay'])

        return loss