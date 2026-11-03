"""Núcleo D2DMoE portado a k2moe (MIT, github.com/bartwojcik/D2DMoE).

Szatkowski et al., "Exploiting Activation Sparsity with Dense to
Dynamic-k Mixture-of-Experts Conversion", NeurIPS 2024.

Adaptado para MLPs gated tipo LLaMA/Gemma/K2-Horizon (gate_proj/up_proj/
down_proj, detección por atributos, no por clase), y con `layer_ids`
para convertir solo X capas en vez de todo el modelo.

Contenido:
  - find_gated_mlps / layer index desde el nombre
  - swap_ffn_activation_to_relu (SiLU/GELU -> ReLU, requisito D2DMoE)
  - MoeficationRouter (router MLP, como en el paper)
  - MoeficationMoE (expertos denso-ponderados; no ahorra FLOPs, sirve
    para entrenar/verificar) + forward_mode 'all' | 'topk' | 'dynk'
  - split_ffn_into_experts (k-means con restricción de tamaño, sobre
    gate_proj normalizado)
  - replace_with_moes(model, num_experts, layer_ids=None)
"""

import re
from typing import Optional, Sequence

import torch
from torch import nn

_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")
_H_RE = re.compile(r"\.h\.(\d+)\.")


class _Abs(nn.Module):
    def forward(self, x):
        return torch.abs(x)


def layer_index_from_name(name: str):
    m = _LAYER_RE.search(name) or _H_RE.search(name)
    return int(m.group(1)) if m else None


def is_gated_mlp(m) -> bool:
    for attr in ("gate_proj", "up_proj", "down_proj"):
        if not hasattr(m, attr):
            return False
        if not isinstance(getattr(m, attr), nn.Linear):
            return False
    return True


def find_gated_mlps(model, layer_ids: Optional[Sequence[int]] = None):
    ids = None if layer_ids is None else set(int(i) for i in layer_ids)
    out = []
    for name, m in model.named_modules():
        if not is_gated_mlp(m):
            continue
        idx = layer_index_from_name(name)
        if ids is not None and idx not in ids:
            continue
        out.append((idx, name, m))
    return out


def swap_ffn_activation_to_relu(model, layer_ids: Optional[Sequence[int]] = None):
    """Cambia la activación de las MLPs gated a ReLU. Devuelve cuántas."""
    n = 0
    attrs = ("act_fn", "act", "activation")
    for idx, name, m in find_gated_mlps(model, layer_ids):
        for a in attrs:
            act = getattr(m, a, None)
            if isinstance(act, nn.Module):
                setattr(m, a, nn.ReLU())
                print(f"  {name}: {type(act).__name__} -> ReLU")
                n += 1
                break
        else:
            print(f"  {name}: NO encuentro atributo de activación, lo salteo")
    return n


class MoeficationRouter(nn.Module):
    """Router MLP como en D2DMoE (scripts DSTI del paper):
      activation='relu' (o 'tanh'), output_activation='abs'.

    Predicción no negativa (Abs al final): predice la norma ℓ2 del output
    de cada experto (regresión, Eq. 3 del paper)."""

    _ACT = {"relu": nn.ReLU, "gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU}

    def __init__(self, hidden_dim, num_experts, width=128, depth=2, bias=False,
                 activation="relu"):
        super().__init__()
        act = self._ACT[activation]
        layers = []
        if depth == 1:
            layers += [nn.Linear(hidden_dim, num_experts, bias=bias)]
        else:
            layers += [nn.Linear(hidden_dim, width, bias=bias), act()]
            for _ in range(depth - 2):
                layers += [nn.Linear(width, width, bias=bias), act()]
            layers += [nn.Linear(width, num_experts, bias=bias)]
        layers += [_Abs()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def add_routers(model, layer_ids=None, width=128, depth=2, bias=False):
    """Añade (o reemplaza) routers en las MoeficationMoE. Devuelve cuántas."""
    ids = None if layer_ids is None else set(int(i) for i in layer_ids)
    n = 0
    for name, m in model.named_modules():
        if not isinstance(m, MoeficationMoE):
            continue
        idx = layer_index_from_name(name)
        if ids is not None and idx not in ids:
            continue
        m.router = MoeficationRouter(m.hidden_dim, m.num_experts,
                                     width=width, depth=depth, bias=bias)
        n += 1
        print(f"  router añadido en {name} (width={width}, depth={depth})")
    return n


class _Expert(nn.Sequential):
    """gate_proj -> act -> down_proj para un trozo de neuronas."""

    def __init__(self, hidden_dim, expert_dim, activation=nn.ReLU,
                 gated=True, bias=False):
        layers = [nn.Linear(hidden_dim, expert_dim, bias=bias), activation()]
        if gated:
            layers += [nn.Linear(hidden_dim, expert_dim, bias=bias), activation()]
        layers += [nn.Linear(expert_dim, hidden_dim, bias=bias)]
        super().__init__(*layers)
        self.gated = gated

    def forward(self, x):
        g = self[0](x)
        if self.gated:
            u = self[2](x)
            return self[-1](torch.nn.functional.relu(g) * u)
        return self[-1](torch.nn.functional.relu(g))


class MoeficationMoE(nn.Module):
    """MoE denso-ponderado: calcula todos los expertos y pondera por el router.

    No ahorra FLOPs (sirve para entrenar/verificar el esquema).
    forward_mode: 'all' (router ignorado) | 'topk' | 'dynk' (umbral por tau)
    """

    def __init__(self, hidden_dim, d_ff, num_experts, activation=nn.ReLU,
                 gated=True, bias=False):
        super().__init__()
        assert d_ff % num_experts == 0, "d_ff debe ser divisible por num_experts"
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.expert_dim = d_ff // num_experts
        self.gated = gated
        self.experts = nn.ModuleList(
            _Expert(hidden_dim, self.expert_dim, activation, gated, bias)
            for _ in range(num_experts))
        self.router = None  # añadir con MoeficationRouter
        self.forward_mode = "all"
        self.k = num_experts       # topk
        self.tau = 0.9             # dynk

    def set_router(self, router: MoeficationRouter):
        self.router = router
        return self

    def gate(self, x):
        b, s, _ = x.shape
        if self.forward_mode == "all" or self.router is None:
            return x.new_ones(b, s, self.num_experts)
        scores = self.router(x)  # [b, s, E]
        if self.forward_mode == "topk":
            top = torch.topk(scores, k=self.k, dim=-1)[1]
            rt = torch.zeros_like(scores)
            rt.scatter_(2, top, 1.0)
            return rt
        if self.forward_mode == "dynk":
            shares = scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            srt, idx = torch.sort(shares, dim=-1, descending=True)
            cum = torch.cumsum(srt, dim=-1)
            mask = (cum < self.tau).roll(shifts=1, dims=-1)
            mask[..., 0] = True
            return mask.gather(-1, idx.argsort(-1)).to(x.dtype)
        raise ValueError(f"forward_mode desconocido: {self.forward_mode}")

    def forward(self, x):
        rt = self.gate(x)  # [b, s, E]
        out = 0.0
        for i, expert in enumerate(self.experts):
            out = out + expert(x) * rt[..., i:i + 1]
        return out, {"routing": rt}


def set_moe_forward_mode(model, mode, k=None, tau=None):
    """Cambia el forward_mode de todas las MoeficationMoE.

    'all'  -> denso-equivalente (equivale a FFN original; usar en entrenamiento)
    'topk' -> top-k por router
    'dynk' -> dynamic-k (Eq. 4 del paper; usar en inferencia)
    """
    n = 0
    for m in model.modules():
        if isinstance(m, MoeficationMoE):
            m.forward_mode = mode
            if k is not None:
                m.k = min(int(k), m.num_experts)
            if tau is not None:
                m.tau = float(tau)
            n += 1
    print(f"forward_mode -> {mode} en {n} capas MoE" +
          (f" (k={k})" if k is not None else f" (tau={tau})" if tau is not None else ""))
    return n


# ---------- router training (paper: routers only, resto congelado) ----------

def _moe_inputs(model, x):
    entries = [(n, m) for n, m in model.named_modules() if isinstance(m, MoeficationMoE)]
    saved = {}
    hooks = []
    for n, m in entries:
        hooks.append(m.register_forward_hook(
            lambda mod, inp, out, key=id(m): saved.setdefault(key, inp[0].detach())))
    model(input_ids=x)
    for h in hooks:
        h.remove()
    return [(n, m, saved.get(id(m))) for n, m in entries]


def train_routers(model, tokens, seq_len, device, steps=500, batch_size=1,
                  lr=1e-3, width=128, depth=2, log_every=25, seed=42,
                  layer_ids=None):
    """Router training estilo D2DMoE: congela el modelo y entrena SOLO routers.

    Loss = MSE(router(x), ||E_i(x)||_2)  (Ec. 3 del paper)
    layer_ids: si se pasa, solo routers de esas capas (los demás congelados).
    """
    moe_entries = [(n, m) for n, m in model.named_modules()
                   if isinstance(m, MoeficationMoE)
                   and (layer_ids is None or layer_index_from_name(n) in set(layer_ids))]
    if not moe_entries:
        print("  train_routers: no hay capas MoE en el modelo")
        return
    p0 = next(model.parameters())
    for n, m in moe_entries:
        if m.router is None:
            m.router = MoeficationRouter(m.hidden_dim, m.num_experts,
                                         width=width, depth=depth)
        m.router.to(device=device, dtype=p0.dtype)
    for p in model.parameters():
        p.requires_grad_(False)
    for _, m in moe_entries:
        for p in m.router.parameters():
            p.requires_grad_(True)
    model.eval()
    opt = torch.optim.AdamW(
        [p for _, m in moe_entries for p in m.router.parameters()], lr=lr)
    g = torch.Generator().manual_seed(seed)
    max_off = max(1, len(tokens) - seq_len - 1)
    print(f"  Router training: {len(moe_entries)} MoE | {steps} steps | lr {lr} | bs {batch_size}")
    for step in range(1, steps + 1):
        offs = [int(torch.randint(0, max_off, (1,), generator=g)) for _ in range(batch_size)]
        x = torch.tensor([tokens[o:o + seq_len] for o in offs], dtype=torch.long, device=device)
        entries = _moe_inputs(model, x)
        losses = []
        for n, m, inp in entries:
            with torch.no_grad():
                targets = torch.stack(
                    [e(inp).pow(2).sum(-1).sqrt() for e in m.experts], dim=-1)
            losses.append(torch.nn.functional.mse_loss(m.router(inp), targets))
        loss = torch.stack(losses).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step == 1 or step % log_every == 0:
            print(f"    router s{step}/{steps} loss {float(loss):.6f}", flush=True)
    model.train()


@torch.no_grad()
def split_ffn_into_experts(ffn, moe: MoeficationMoE):
    """Reparte las neuronas del FFN en expertos por k-means restringido
    sobre gate_proj normalizado. Si falta k-means-constrained, intenta
    instalarlo con pip; si no, falla con instrucción clara."""
    try:
        from k_means_constrained import KMeansConstrained
    except ImportError:
        import subprocess
        import sys
        print("  k-means-constrained no está; instalando con pip...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                                   "k-means-constrained"])
            from k_means_constrained import KMeansConstrained
        except Exception as e:
            raise ImportError(
                "Falta k-means-constrained. Instálalo en Colab con: "
                "!pip install k-means-constrained") from e
    w = ffn.gate_proj.weight  # [d_ff, hidden]
    labels = KMeansConstrained(n_clusters=moe.num_experts,
                               size_min=moe.expert_dim,
                               size_max=moe.expert_dim) \
        .fit_predict(torch.nn.functional.normalize(w, p=2, dim=1).cpu().numpy())
    for e in range(moe.num_experts):
        idx = [i for i, l in enumerate(labels) if l == e]
        idx_t = torch.tensor(idx, dtype=torch.long, device=w.device)
        dst = moe.experts[e]
        dst[0].weight.copy_(ffn.gate_proj.weight[idx_t])   # gate
        dst[2].weight.copy_(ffn.up_proj.weight[idx_t])     # up
        dst[-1].weight.copy_(ffn.down_proj.weight[:, idx_t])  # down (columnas)
        if ffn.gate_proj.bias is not None:
            dst[0].bias.copy_(ffn.gate_proj.bias[idx_t])
            dst[2].bias.copy_(ffn.up_proj.bias[idx_t])
    return labels


def replace_with_moes(model, num_experts: int, layer_ids: Optional[Sequence[int]] = None,
                      activation=nn.ReLU, split_weights: bool = True):
    """Sustituye MLPs gated por MoeficationMoE (todas o solo layer_ids).

    split_weights=False salta el k-means (los pesos vienen de un checkpoint,
    p.ej. al retomar). Devuelve los nombres reemplazados."""
    import torch.nn as _nn
    replaced = []
    for idx, name, ffn in find_gated_mlps(model, layer_ids):
        hidden, d_ff = ffn.gate_proj.in_features, ffn.gate_proj.out_features
        ref_param = next(ffn.parameters())
        moe = MoeficationMoE(hidden, d_ff, num_experts,
                             activation=activation,
                             gated=True, bias=ffn.gate_proj.bias is not None)
        moe.to(device=ref_param.device, dtype=ref_param.dtype)
        if split_weights:
            with torch.no_grad():
                split_ffn_into_experts(ffn, moe)
        parent_name, attr = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name) if parent_name else model
        assert isinstance(parent, _nn.Module)
        setattr(parent, attr, moe)
        replaced.append(name)
        print(f"  {name} -> MoE ({num_experts} expertos x {moe.expert_dim})")
    return replaced
