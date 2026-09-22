"""PRiSM para rebuild: seleccion training-free de capas + freeze.

Idea (LREC 2026): por cada batch se agrega un vector por capa
(h_agg = ultimo token no-pad en causales), Score(l) = coseno medio
contra TODAS las demas capas. Se promedia sobre batches del dataset,
se parte en k bloques contiguos y se elige 1 capa por bloque (argmax).
Solo esas capas entrenan; el resto va con requires_grad=False.

Uso con K2-Horizon-0.9B (causal):
    selected = run_prism_selection(model, token_ids, k=4, seq_len=2048, device, pad_id)
    freeze_except_layers(model, selected)
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Optional, Callable
import torch
import torch.nn.functional as F


@torch.no_grad()
def aggregate_last_token(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Un vector por capa y ejemplo: ultimo token no-pad. [B, L, D]."""
    reps = []
    for h in hidden_states:
        if attention_mask is None:
            reps.append(h[:, -1, :])
        else:
            lengths = attention_mask.long().sum(dim=1).clamp_min(1)
            idx = lengths - 1
            batch_idx = torch.arange(h.size(0), device=h.device)
            reps.append(h[batch_idx, idx, :])
    return torch.stack(reps, dim=1)


@torch.no_grad()
def prism_scores_for_batch(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Score(l) = mean_{j != l} cosine(h_l, h_j), promediado sobre el batch. [L]."""
    reps = F.normalize(aggregate_last_token(hidden_states, attention_mask), p=2, dim=-1)
    sim = torch.matmul(reps, reps.transpose(1, 2))  # [B, L, L]
    L = sim.size(1)
    if L < 2:
        raise ValueError("PRiSM requiere al menos 2 capas.")
    mask = ~torch.eye(L, dtype=torch.bool, device=sim.device)
    off_diag = sim[:, mask].reshape(sim.size(0), L, L - 1)
    return off_diag.mean(dim=-1).mean(dim=0)  # [L]


def get_transformer_layers(model, layer_getter: Optional[Callable] = None):
    """Ubicaciones comunes de capas en modelos HF + custom (K2Horizon)."""
    if layer_getter is not None:
        return layer_getter(model)
    candidates = [
        lambda m: m.model.layers,            # Llama / Gemma / K2Horizon-like
        lambda m: m.language_model.layers,
        lambda m: m.transformer.h,           # GPT-2
        lambda m: m.encoder.layer,           # BERT-like
        lambda m: m.decoder.layers,
    ]
    for getter in candidates:
        try:
            return getter(model)
        except AttributeError:
            pass
    raise ValueError("No encontre las capas del Transformer. Pasa layer_getter().")


def contiguous_blocks(num_layers: int, k: int) -> List[List[int]]:
    """L capas en k bloques contiguos sin solaparse (resto a los primeros)."""
    if not (1 <= k <= num_layers):
        raise ValueError(f"Requiere 1 <= k <= L, k={k}, L={num_layers}")
    q, r = divmod(num_layers, k)
    blocks, start = [], 0
    for i in range(k):
        size = q + (1 if i < r else 0)
        blocks.append(list(range(start, start + size)))
        start += size
    return blocks


def select_prism_layers(scores: torch.Tensor, k: int) -> List[int]:
    """Un argmax por bloque contiguo. Indices 0-based."""
    selected = []
    for block in contiguous_blocks(scores.numel(), k):
        best = int(torch.argmax(scores[block]).item())
        selected.append(block[best])
    return selected


def freeze_except_layers(model, selected_layers: Sequence[int],
                         layer_getter: Optional[Callable] = None):
    """requires_grad=False en todo el backbone menos las elegidas.

    Embeddings, norms finales y lm_head quedan como estan (entrenables).
    """
    selected = set(int(x) for x in selected_layers)
    for i, layer in enumerate(get_transformer_layers(model, layer_getter)):
        for p in layer.parameters():
            p.requires_grad = (i in selected)
    return model


def trainable_parameter_count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_prism_result(scores: torch.Tensor, selected: Sequence[int]):
    print("PRiSM scores por capa:")
    for i, s in enumerate(scores.tolist()):
        mark = "  <-- ENTRENA" if i in selected else ""
        print(f"  capa {i:3d}: {s:.6f}{mark}")


def build_batches(token_ids: List[int], seq_len: int, batch_size: int,
                  pad_id: int, device: torch.device, n_batches: int = 4):
    """Corta seqs de seq_len de la lista de ids y arma dicts con attention_mask."""
    need = n_batches * batch_size * seq_len
    ids = list(token_ids[:need])
    ids += [pad_id] * max(0, need - len(ids))
    batches = []
    for b in range(n_batches):
        xs, ms = [], []
        for j in range(batch_size):
            s = ids[(b * batch_size + j) * seq_len:(b * batch_size + j + 1) * seq_len]
            xs.append(torch.tensor(s, dtype=torch.long))
            ms.append(torch.tensor([1 if t != pad_id else 0 for t in s], dtype=torch.long))
        batches.append({"input_ids": torch.stack(xs).to(device),
                        "attention_mask": torch.stack(ms).to(device)})
    return batches


@torch.no_grad()
def _hidden_states_via_hooks(model, layers, input_ids, attention_mask):
    """Captura la salida de cada capa con hooks (para forwards que no
    devuelven hidden_states, como K2Horizon)."""
    outs = []
    handles = [l.register_forward_hook(
        lambda m, i, o: outs.append(o if torch.is_tensor(o) else o[0]))
        for l in layers]
    try:
        model(input_ids=input_ids, attention_mask=attention_mask,
              return_dict=True)
    finally:
        for h in handles:
            h.remove()
    return outs


@torch.no_grad()
def run_prism_selection(model, token_ids, k: int, seq_len: int,
                        device: torch.device, pad_id: int,
                        n_batches: int = 4, batch_size: int = 2,
                        layer_getter: Optional[Callable] = None) -> List[int]:
    """Pipeline completo: batches del dataset -> scores -> top-1 por bloque."""
    was_training = model.training
    model.eval()
    layers = get_transformer_layers(model)
    print(f"PRiSM: {len(layers)} capas, {n_batches} batches calibracion...", flush=True)
    batches = build_batches(token_ids, seq_len, batch_size, pad_id, device, n_batches)
    total, n = None, 0
    for bi, batch in enumerate(batches):
        print(f"  PRiSM batch {bi + 1}/{len(batches)}...", flush=True)
        out = model(input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    output_hidden_states=True, return_dict=True)
        hs = getattr(out, "hidden_states", None)
        if hs is None:
            # Forward custom que no devuelve hidden_states: hooks por capa.
            hs = [None] + _hidden_states_via_hooks(
                model, get_transformer_layers(model),
                batch["input_ids"], batch["attention_mask"])
        bs = prism_scores_for_batch(hs[1:], batch["attention_mask"])
        total = bs if total is None else total + bs
        n += 1
    scores = total / max(n, 1)
    selected = select_prism_layers(scores, k)
    print_prism_result(scores, selected)
    if was_training:
        model.train()
    return selected
