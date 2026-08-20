import math
import torch
from torch.optim.optimizer import Optimizer


def _newton_schulz(G, steps=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    orig_dtype = G.dtype
    G = G.float()
    X = G / (G.norm() + 1e-7)
    transposed = G.shape[0] > G.shape[1]
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(orig_dtype)


class Muon(Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, weight_decay=0.01, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]

                if "mu" not in state:
                    state["mu"] = torch.zeros_like(p)

                if grad.ndim in (2, 3):
                    grad_ortho = _newton_schulz(grad, steps=group.get("ns_steps", 5))
                else:
                    grad_ortho = grad

                mu = state["mu"]
                mu.mul_(group["momentum"]).add_(grad_ortho)
                update = grad_ortho + group["momentum"] * mu

                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                p.add_(update, alpha=-group["lr"])
        return loss


def get_param_groups(model, muon_lr=0.02, adam_lr=3e-4, weight_decay=0.01):
    muon_params = []
    adam_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "weight" in name and param.ndim in (2, 3):
            muon_params.append(param)
        else:
            adam_params.append(param)
    return [
        {"params": muon_params, "lr": muon_lr, "weight_decay": weight_decay},
        {"params": adam_params, "lr": adam_lr, "weight_decay": 0.0},
    ]


def wsd_schedule(peak_value, total_steps, warmup_steps, decay_ratio=0.15):
    decay_steps = max(1, int(total_steps * decay_ratio))
    stable_steps = total_steps - warmup_steps - decay_steps

    def schedule(step):
        if step < warmup_steps:
            return peak_value * step / max(warmup_steps, 1)
        elif step < warmup_steps + stable_steps:
            return peak_value
        else:
            progress = (step - warmup_steps - stable_steps) / max(decay_steps, 1)
            return peak_value * 0.5 * (1 + math.cos(math.pi * progress))

    return schedule
