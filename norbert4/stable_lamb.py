import torch


class StableLamb(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6, weight_decay=0, cautious=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, cautious=cautious)
        super(StableLamb, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = True
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Lamb does not support sparse gradients, consider SparseAdam instad.')

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                m_t = exp_avg / bias_correction1
                v_t = exp_avg_sq / bias_correction2

                # compute per tensor RMS stabilization term
                rms = grad.pow(2).div_(v_t + group['eps']).mean().sqrt()
                torch.sqrt_(v_t)

                update = m_t / ((v_t + group['eps']) * max(1.0, rms.item()))

                if group['cautious']:
                    caution = (m_t * grad > 0).to(grad.dtype)
                    caution.div_(caution.mean() + group['eps'])
                    update.mul_(caution)

                ratio = 1.0
                if group['weight_decay'] > 0:
                    update.add_(p.data, alpha=group['weight_decay'])

                    g_norm = torch.norm(update.flatten())
                    w_norm = torch.norm(p.data.flatten())

                    if w_norm > 0.0 and g_norm > 0.0:
                        ratio = w_norm / g_norm

                p.data.add_(update, alpha=-group['lr'] * ratio)

        return loss
