"""MoSE de ANCHO para K2 denso: router por capa elige el ancho slimmable del FF.

K2-0.9B es denso (sin expertos): el router elige SOLO el ancho
(25/50/75/100%) del gate/up/down EXISTENTES de K2. No se toca
atencion ni RoPE. El train ajusta el router en cada step via
z-loss + load-balance sumadas al loss (ver collect_aux_loss).

Forward devuelve (out, aux) como el MoE nativo: el decoder de K2 ya
desempaqueta tuplas del mlp, el aux se suma en el loop de train.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseWidthMoSE(nn.Module):
    """Router de ancho sobre un FF SwiGLU denso ya existente."""

    def __init__(self, gate_proj, up_proj, down_proj,
                 widths=(0.25, 0.50, 0.75, 1.0),
                 z_loss_gamma=0.0001, load_balance_gamma=0.0001,
                 bias_decay=0.1, noise_std=0.005):
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self.d_model = gate_proj.in_features
        self.inter = gate_proj.out_features
        self.widths = tuple(widths)
        self.n_widths = len(self.widths)
        self.z_loss_gamma = z_loss_gamma
        self.load_balance_gamma = load_balance_gamma
        self.bias_decay = bias_decay
        self.noise_std = noise_std

        # Router: un score por ancho (SOLO ancho, sin expertos).
        self.router = nn.Linear(self.d_model, self.n_widths, bias=False)
        nn.init.normal_(self.router.weight, std=0.02)
        self.router.to(gate_proj.weight.device)
        self.register_buffer("route_bias", torch.zeros(self.n_widths))
        self.register_buffer("last_counts", torch.zeros(self.n_widths, dtype=torch.long))
        self._last_aux_loss = None  # aux CON grafo (el train lo suma al loss)
        self.force_full = False  # train Eq.(6) pass 2: ejecuta full pero el aux ajusta igual

    def forward(self, x):
        B, T, C = x.shape
        # Capa congelada (PRiSM no la eligio): FF denso EXACTO al 100%,
        # sin router (el router sin entrenar elegiria anchos al azar).
        if not self.router.weight.requires_grad:
            h = F.silu(F.linear(x, self.gate_proj.weight, None))
            h = h * F.linear(x, self.up_proj.weight, None)
            out = F.linear(h, self.down_proj.weight, None)
            self._last_aux_loss = None
            return out, torch.tensor(0.0, device=x.device)
        N = B * T
        xf = x.reshape(N, C)

        logits = self.router(xf)
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        biased = logits + self.route_bias.to(logits.dtype)
        probs = F.softmax(biased, dim=-1)
        top_i = probs.argmax(dim=-1)

        # Aux para ajustar el router en cada step (con gradiente).
        if self.z_loss_gamma > 0:
            lse = torch.logsumexp(logits, dim=-1)
            z_loss = self.z_loss_gamma * (lse ** 2).mean()
        else:
            z_loss = torch.tensor(0.0, device=x.device)
        if self.load_balance_gamma > 0:
            p_mean = probs.mean(dim=0)
            target = torch.full_like(p_mean, 1.0 / float(self.n_widths))
            lb_loss = self.load_balance_gamma * ((p_mean - target) ** 2).sum()
        else:
            lb_loss = torch.tensor(0.0, device=x.device)
        aux_loss = z_loss + lb_loss

        with torch.no_grad():
            counts = torch.bincount(top_i, minlength=self.n_widths)
            self.last_counts = counts.clone()
            ntk = max(N, 1)
            delta = self.bias_decay * (N / self.n_widths - counts.float()) / ntk
            self.route_bias.add_(delta.to(self.route_bias.dtype))

        self._last_aux_loss = aux_loss

        # Pass 2 Eq.(6): ejecuta full pero el aux ajusta el router igual.
        if self.force_full:
            h = F.silu(F.linear(x, self.gate_proj.weight, None))
            h = h * F.linear(x, self.up_proj.weight, None)
            return F.linear(h, self.down_proj.weight, None), aux_loss

        # Pass 1: top-1 ancho por token (seleccion dura, peso 1.0).
        out = torch.empty_like(xf)
        for wii in range(self.n_widths):
            idx = (top_i == wii).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            d = max(8, int(self.inter * self.widths[wii]))
            g = F.linear(xf[idx], self.gate_proj.weight[:d, :], None)
            u = F.linear(xf[idx], self.up_proj.weight[:d, :], None)
            h = F.silu(g) * u
            out[idx] = F.linear(h, self.down_proj.weight[:, :d], None)
        out = out.reshape(B, T, C)
        return out, aux_loss

    def width_str(self):
        total = int(self.last_counts.sum().item()) or 1
        return " ".join(
            f"{int(w * 100)}%:{int(self.last_counts[i].item() * 100 // total)}%"
            for i, w in enumerate(self.widths))


def wrap_k2_with_mose(model, widths=(0.25, 0.50, 0.75, 1.0), **kw):
    """Reemplaza el mlp denso de cada capa K2 por DenseWidthMoSE (mismos pesos)."""
    layers = model.model.layers
    n = 0
    for layer in layers:
        mlp = layer.mlp
        if isinstance(mlp, DenseWidthMoSE):
            continue
        if not (hasattr(mlp, "gate_proj") and hasattr(mlp, "up_proj")
                and hasattr(mlp, "down_proj")):
            continue  # bloque MoE nativo u otro: no tocar
        layer.mlp = DenseWidthMoSE(
            mlp.gate_proj, mlp.up_proj, mlp.down_proj, widths=widths, **kw)
        n += 1
    print(f"MoSE-ancho: {n}/{len(layers)} capas convertidas")
    return model


def set_force_full(model, v: bool):
    """Activa/desactiva pass full en todos los routers (Eq.6)."""
    for m in model.modules():
        if isinstance(m, DenseWidthMoSE):
            m.force_full = v


def collect_aux_loss(model):
    """Suma el aux (con grafo) de todos los routers. 0.0 si no hay."""
    total = None
    for m in model.modules():
        if isinstance(m, DenseWidthMoSE) and m._last_aux_loss is not None:
            total = m._last_aux_loss if total is None else total + m._last_aux_loss
    if total is None:
        try:
            dev = next(model.parameters()).device
        except StopIteration:
            dev = torch.device("cpu")
        return torch.tensor(0.0, device=dev)
    return total
