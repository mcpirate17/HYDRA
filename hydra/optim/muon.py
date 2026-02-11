"""
Muon: Momentum Orthogonalized Optimizer.
References: 
- https://github.com/KellerJordan/Muon
- https://arxiv.org/abs/2410.09331
"""
import torch
import torch.distributed as dist

@torch.compile
def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 10, eps: float = 1e-7) -> torch.Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power of the matrix G.
    Approximates UV^T where G = USV^T.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    
    # Ensure spectral norm < sqrt(4/3)
    X /= (X.norm() + eps) 
    
    if G.size(0) > G.size(1):
        X = X.T
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    if G.size(0) > G.size(1):
        X = X.T
        
    return X

class Muon(torch.optim.Optimizer):
    """
    Muon - Momentum Orthogonalized Optimizer.
    
    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-processing step.
    This effectively whitens the updates.
    
    1D/scalar params are handled by a fused AdamW instance (single CUDA kernel per step).
    2D params (>1024 elements) use Newton-Schulz orthogonalization.
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, adamw_params=None, adamw_lr=3e-4, adamw_betas=(0.9, 0.95), adamw_eps=1e-8, adamw_wd=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        
        # Segregate params into 2D (Muon) and other (AdamW)
        params = list(params)
        muon_params = [p for p in params if p.ndim == 2 and p.numel() > 1024] # Heuristic: only 2D and >1k params
        adamw_param_list = [p for p in params if p.ndim != 2 or p.numel() <= 1024]
        
        super().__init__(muon_params, defaults)
        
        # Store param sets for easy access
        self.muon_params = set(muon_params)
        self.adamw_params = set(adamw_param_list)
        
        # Use PyTorch's fused AdamW for 1D/scalar params — single CUDA kernel per step
        # instead of 7 separate kernel launches per parameter.
        if adamw_param_list:
            use_fused = torch.cuda.is_available()
            self._adamw = torch.optim.AdamW(
                adamw_param_list,
                lr=adamw_lr,
                betas=adamw_betas,
                eps=adamw_eps,
                weight_decay=adamw_wd,
                fused=use_fused,
            )
        else:
            self._adamw = None

    def zero_grad(self, set_to_none: bool = True) -> None:
        """Zero gradients for both Muon (2D) and AdamW (1D/scalar) params."""
        super().zero_grad(set_to_none=set_to_none)
        if self._adamw is not None:
            self._adamw.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        """Return combined state dict for both Muon and internal AdamW."""
        sd = super().state_dict()
        if self._adamw is not None:
            sd['_adamw_state_dict'] = self._adamw.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        """Load combined state dict for both Muon and internal AdamW."""
        adamw_sd = state_dict.pop('_adamw_state_dict', None)
        super().load_state_dict(state_dict)
        if adamw_sd is not None and self._adamw is not None:
            self._adamw.load_state_dict(adamw_sd)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        # Step the fused AdamW for 1D/scalar params
        if self._adamw is not None:
            self._adamw.step()

        # Step Muon for 2D params — batched momentum via _foreach_ ops
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            # Collect active params/grads/buffers
            params, grads, bufs = [], [], []
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                params.append(p)
                grads.append(p.grad)
                bufs.append(state['momentum_buffer'])

            if not params:
                continue

            # Batched momentum update: buf = momentum * buf + grad
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads)

            # Nesterov: g = grad + momentum * buf; else g = buf
            if nesterov:
                gs = [grad + momentum * buf for grad, buf in zip(grads, bufs)]
            else:
                gs = list(bufs)

            # Newton-Schulz orthogonalization (per-param, variable shapes)
            updates = []
            for g in gs:
                u = zeropower_via_newtonschulz5(g, steps=ns_steps)
                u.mul_(max(1, u.size(0) / u.size(1)) ** 0.5)
                updates.append(u)

            # Batched param update: p -= lr * update
            torch._foreach_add_(params, updates, alpha=-lr)

        return loss
