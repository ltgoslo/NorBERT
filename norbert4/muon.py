import torch
import torch.distributed as dist
import os
import math
from torch import Tensor

# -----------------------------------------------------------------------------
# Muon optimizer

@torch.compile()
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2
    # a = [3.6, 3.6, 3.6, 3.6, 3.6 / 1.4]
    # b = [-6.77, -7.43, -8.14, -8.2, -4.27/1.4]
    # c = [3.69, 4.42, 5.33, 5.71, 2.1 / 1.4]
    # a, b, c = (3.4445, -4.7750,  2.0315)
    a = [3.9641, 3.8994, 3.7125, 3.5012, 2.6343, 2.2590]
    b = [-8.1733, -7.9979, -7.4824, -6.7629, -3.9431, -2.1785]
    c = [4.3135, 4.2336, 3.9957, 3.717, 2.1477, 0.9395]
    # a = [4.0848, 3.9505, 3.7418, 2.8769, 2.8366]
    # b = [-6.8946, -6.3029, -5.5913, -3.1427, -3.0525]
    # c = [2.9270, 2.6377, 2.3037, 1.2046, 1.2012]
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for i in range(steps):
        A = X @ X.mT
        B = b[i] * A + c[i] * A @ A  # adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a[i] * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    
    # X = torch.einsum('ij,ij,ab->ab', G.type_as(X), X, X)  # Adaptive scaling,`(G * X).sum() * X` == (G.T @ X).trace() * X
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.
    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).
    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iteration steps to use.
    """
    def __init__(self, params, lr=0.02, weight_decay=0.01, momentum=0.95, nesterov=True, ns_steps=6, rank=0, world_size=1):
        self.world_size = world_size
        self.rank = rank
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        params: list[torch.Tensor] = [*params]
        assert all(isinstance(p, torch.Tensor) for p in params)
        sizes = {p.numel() for p in params}
        
        def create_update_buffer(size:int):
            b = torch.empty(self.world_size, size, dtype=torch.bfloat16, device="cuda")
            return dict(update_buffer=b, update_buffer_views=[b[i] for i in range(self.world_size)])
        
        param_groups = [
                dict(params=[p for p in params if p.numel() == size], **create_update_buffer(size)) for size in sizes]
        super().__init__(param_groups, defaults)
        # pre-init momentum
        # for group in self.param_groups:
        #     for p in group['params']:
        #         self.state[p]['momentum_buffer'] = torch.zeros(p.shape, device=p.device, dtype=torch.float)

    def adjust_lr_for_muon(self, lr, param_shape):
        A, B = param_shape[:2]
        # We adjust the learning rate and weight decay based on the size of the parameter matrix
        # as describted in the paper
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
        adjusted_lr = lr * adjusted_ratio
        return adjusted_lr

    @torch.no_grad()
    def step(self):

        for group in self.param_groups:

            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            update_buffers = group['update_buffer']
            update_buffer_views: list[Tensor] = group["update_buffer_views"]
            # generate weight updates in distributed fashion
            params: list[torch.Tensor] = group['params']
            handle = None
            params_world = None

            def update_prev():
                if params_world is None:
                    return
                assert handle is not None
                handle.wait()
                for p_world, g_world in zip(params_world, update_buffer_views):
                    p_world.mul_(1 - group["lr"] * group["weight_decay"])
                    adjusted_lr = self.adjust_lr_for_muon(lr, p_world.shape)
                    p_world.add_(
                        g_world.view_as(p_world),
                        # alpha=-lr * max(1, p_world.size(-2) / p_world.size(-1)) ** 0.5,
                        alpha=-adjusted_lr
                    )
            for base_i in range(len(params))[::self.world_size]:
                if base_i + self.rank < len(params):
                    p = params[base_i + self.rank]
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf: torch.Tensor = state['momentum_buffer']
                    buf.lerp_(g, 1 - momentum)
                    g = g.lerp_(buf, momentum) if nesterov else buf
                    g = zeropower_via_newtonschulz5(g, steps=ns_steps).flatten()
                else:
                    g = update_buffer_views[self.rank]
                update_prev()  # async all_gather instead of sync all_reduce by @YouJiacheng
                handle = dist.all_gather_into_tensor(update_buffers, g, async_op=True)
                params_world = params[base_i : base_i + self.world_size]
            update_prev()
