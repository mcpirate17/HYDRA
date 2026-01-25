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
    """
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, adamw_params=None, adamw_lr=3e-4, adamw_betas=(0.9, 0.95), adamw_eps=1e-8, adamw_wd=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_lr_ratio=adamw_lr/lr, 
                        adamw_betas=adamw_betas, 
                        adamw_eps=adamw_eps, 
                        adamw_wd=adamw_wd)
        
        # Segregate params into 2D (Muon) and other (AdamW)
        params = list(params)
        muon_params = [p for p in params if p.ndim == 2 and p.numel() > 1024] # Heuristic: only 2D and >1k params
        adamw_params = [p for p in params if p.ndim != 2 or p.numel() <= 1024]
        
        super().__init__(muon_params + adamw_params, defaults)
        
        # Store param groups for easy access
        self.muon_params = set(muon_params)
        self.adamw_params = set(adamw_params)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            
            # AdamW sub-parameters
            adamw_lr = lr * group['adamw_lr_ratio']
            beta1, beta2 = group['adamw_betas']
            eps = group['adamw_eps']
            wd = group['adamw_wd']

            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                if 'step' not in state:
                    state['step'] = 0
                
                state['step'] += 1
                
                if p in self.adamw_params:
                    # Standard AdamW
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                        
                    exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                    
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    denom = exp_avg_sq.sqrt().add_(eps)
                    
                    # Weight decay
                    p.mul_(1 - adamw_lr * wd)
                    
                    # Update
                    step_size = adamw_lr
                    if state['step'] < 1000: # Simple warmup
                         step_size *= state['step'] / 1000
                         
                    p.addcdiv_(exp_avg, denom, value=-step_size)
                    
                elif p in self.muon_params:
                    # Muon
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(p)
                    
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    
                    if nesterov:
                        g = grad + momentum * buf
                    else:
                        g = buf
                    
                    # Orthogonalize
                    update = zeropower_via_newtonschulz5(g, steps=ns_steps)
                    
                    # Scale
                    update.mul_(max(1, update.size(0)/update.size(1))**0.5)
                    
                    # Apply
                    p.add_(update, alpha=-lr)

        return loss
