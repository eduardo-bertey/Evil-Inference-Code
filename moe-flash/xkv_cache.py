"""xKV-SR: cross-layer SVD KV cache + selective reconstruction for MLA.

Compresses K/V of `group_size` consecutive layers into a shared low-rank
subspace (rank_k / rank_v). During decode only `sparse_budget` tokens are
reconstructed (chosen via chunk landmarks), plus a small local window.

MLA note: latent C_KV (d_c=64) is unchanged; SVD runs on expanded K/V
after W_up_kv so rank can be 128+ independently of the latent size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass
class XKVConfig:
    group_size: int = 4
    rank_k: int = 128
    rank_v: int = 128
    sparse_budget: int = 2048
    chunk_size: int = 8
    local_window: int = 32
    outlier_budget: int | None = None
    enabled: bool = True
    # Train bajo compresion: fake SVD por capa (con grad) en el forward.
    # Cross-layer solo existe en inferencia (prefill post-hoc, sin grad).
    train_fake_svd: bool = False

    def __post_init__(self):
        if self.sparse_budget % self.chunk_size != 0:
            raise ValueError(
                f"sparse_budget ({self.sparse_budget}) must be divisible by "
                f"chunk_size ({self.chunk_size})"
            )
        if self.outlier_budget is None:
            self.outlier_budget = max(self.chunk_size, (self.sparse_budget // 1024) * 24)
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")


def _fast_svd(tensor: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Low-rank SVD: returns (U [b,s,r], SVh [b,r,d])."""
    orig_dtype = tensor.dtype
    U, S, Vh = torch.linalg.svd(tensor.float(), full_matrices=False)
    r = min(int(rank), int(S.shape[-1]))
    U_t = U[:, :, :r]
    S_t = S[:, :r]
    Vh_t = Vh[:, :r, :]
    SVh = torch.matmul(torch.diag_embed(S_t), Vh_t)
    return U_t.to(orig_dtype), SVh.to(orig_dtype)


def fake_svd(tensor: torch.Tensor, rank: int) -> torch.Tensor:
    """Truncate SVD and multiply back (train-time compression)."""
    U, SVh = _fast_svd(tensor, rank)
    return torch.matmul(U, SVh).to(tensor.dtype)


class XVKSRCoordinator:
    """Cross-layer SVD + landmark retrieval shared across a group of layers.

    Prefill (per layer, in order):
        coord.observe_prefill(layer_idx, k_state, v_state, k_rope)
        # at group end SVD runs automatically
        coord.build_landmarks(layer_idx, k_rope, query_last)

    Decode:
        pos = coord.retrieve(layer_idx, q)
        k_sel, v_sel = coord.reconstruct(layer_idx, pos, rope_fn)
    """

    def __init__(self, cfg: XKVConfig, num_layers: int, num_kv_groups: int,
                 head_dim: int, device: torch.device, dtype: torch.dtype):
        self.cfg = cfg
        self.num_layers = num_layers
        self.nkv = num_kv_groups
        self.hd = head_dim
        self.device = device
        self.dtype = dtype

        self.group_size = cfg.group_size
        self.rank_k = cfg.rank_k
        self.rank_v = cfg.rank_v
        self.sparse_budget = cfg.sparse_budget
        self.chunk_size = cfg.chunk_size
        self.local_window = cfg.local_window
        self.outlier_budget = cfg.outlier_budget
        self.select_sets = cfg.sparse_budget // cfg.chunk_size

        self.U_k: dict[int, torch.Tensor] = {}
        self.SV_k: dict[int, torch.Tensor] = {}
        self.U_v: dict[int, torch.Tensor] = {}
        self.U_v_shared: dict[int, torch.Tensor] = {}  # same U for group start key
        self.SV_v: dict[int, torch.Tensor] = {}

        # store group-shared U under every layer in the group
        self._k_landmark: dict[int, torch.Tensor] = {}
        self._k_landmark_idx: dict[int, torch.Tensor] = {}

        self._k_group: list[torch.Tensor] = []
        self._v_group: list[torch.Tensor] = []
        self._k_rope_group: list[torch.Tensor] = []
        self._group_layers: list[int] = []

        self.seq_len = 0
        self.prefilled = False
        self.chunks = 0
        self.prefill_local = 0
        self.sparse_start = 0
        self.sparse_end = 0
        self._last_pos: torch.Tensor | None = None

    def clear(self) -> None:
        """Reset all SVD factors + landmarks (new prefill / new generate)."""
        self.U_k.clear()
        self.SV_k.clear()
        self.U_v.clear()
        self.SV_v.clear()
        self._k_landmark.clear()
        self._k_landmark_idx.clear()
        self._k_group = []
        self._v_group = []
        self._k_rope_group = []
        self._group_layers = []
        self.seq_len = 0
        self.prefilled = False
        self.chunks = 0
        self.prefill_local = 0
        self.sparse_start = 0
        self.sparse_end = 0
        self._last_pos = None

    # ── Prefill ──────────────────────────────────────────────────────────

    @torch.no_grad()
    def observe_prefill(self, layer_idx: int, k_state: torch.Tensor,
                        v_state: torch.Tensor, k_rope: torch.Tensor) -> None:
        """Accumulate expanded K/V; run SVD when the group is complete.

        k_state/v_state: [b, s, nkv, hd]
        k_rope:          [b, s, nkv|1, hd] post-RoPE keys
        """
        b, s = k_state.shape[0], k_state.shape[1]
        self._k_group.append(k_state.reshape(b, s, -1))
        self._v_group.append(v_state.reshape(b, s, -1))
        if k_rope.dim() == 4 and k_rope.shape[2] == 1:
            k_rope = k_rope.expand(-1, -1, self.nkv, -1)
        self._k_rope_group.append(k_rope)
        self._group_layers.append(layer_idx)

        group_full = (
            len(self._group_layers) == self.group_size
            or layer_idx == self.num_layers - 1
        )
        if group_full:
            self._run_group_svd()

    @torch.no_grad()
    def _run_group_svd(self) -> None:
        if not self._k_group:
            return
        combined_k = torch.cat(self._k_group, dim=2)  # [b, s, nkv*hd*gs]
        combined_v = torch.cat(self._v_group, dim=2)
        U_k, SVh_k = _fast_svd(combined_k, self.rank_k)
        U_v, SVh_v = _fast_svd(combined_v, self.rank_v)

        b = U_k.shape[0]
        n_layers = len(self._group_layers)
        r_k = U_k.shape[-1]
        r_v = U_v.shape[-1]
        # [b, r, nkv*hd*gs] → [b, nkv*gs, r, hd]
        sv_k = SVh_k.view(b, r_k, self.nkv * n_layers, self.hd).transpose(1, 2)
        sv_v = SVh_v.view(b, r_v, self.nkv * n_layers, self.hd).transpose(1, 2)

        for i, li in enumerate(self._group_layers):
            self.U_k[li] = U_k
            self.U_v[li] = U_v
            self.SV_k[li] = sv_k[:, i * self.nkv:(i + 1) * self.nkv].contiguous()
            self.SV_v[li] = sv_v[:, i * self.nkv:(i + 1) * self.nkv].contiguous()

        self._k_group = []
        self._v_group = []
        self._k_rope_group = []
        self._group_layers = []

    @torch.no_grad()
    def build_landmarks(self, layer_idx: int, k_rope: torch.Tensor,
                        query_last: torch.Tensor | None = None) -> torch.Tensor | None:
        """Chunk landmarks + outliers; optional first retrieve with last query.

        k_rope: [b, s, nkv, hd] post-RoPE keys
        query_last: [b, 1, nqh, hd] or None
        Returns position_ids [b, nkv, sparse_budget] or None.
        """
        b, s = k_rope.shape[0], k_rope.shape[1]
        self.seq_len = s
        chunks_total = s // self.chunk_size
        n_local_chunks = max(1, self.local_window // self.chunk_size)

        if chunks_total <= n_local_chunks + 1:
            self.chunks = 0
            self.prefill_local = s
            self.sparse_start = s
            self.sparse_end = s
            self.prefilled = True
            if query_last is not None:
                return self.retrieve(layer_idx, query_last)
            return None

        self.chunks = chunks_total - n_local_chunks
        self.chunks = self.chunks - (self.chunks % 2)
        if self.chunks < 1:
            self.chunks = 1

        self.prefill_local = s - self.chunks * self.chunk_size

        ctx = k_rope[:, : self.chunks * self.chunk_size]
        ctx = ctx.reshape(b, self.nkv, self.chunks, self.chunk_size, self.hd)
        landmark_cand = ctx.mean(dim=3)  # [b, nkv, chunks, hd]

        cos = torch.nn.functional.cosine_similarity(
            landmark_cand.unsqueeze(3).expand(-1, -1, -1, self.chunk_size, -1),
            ctx, dim=-1,
        )  # [b, nkv, chunks, chunk_size]
        outlier_n = min(
            max(1, self.outlier_budget // self.chunk_size),
            max(1, self.chunks // 4),
        )
        outlier_idx = cos.min(dim=-1).values.topk(outlier_n, largest=False).indices

        all_idx = (
            torch.arange(self.chunks, device=k_rope.device)
            .view(1, 1, -1)
            .expand(b, self.nkv, -1)
        )
        mask = torch.ones_like(all_idx, dtype=torch.bool)
        mask.scatter_(-1, outlier_idx, False)
        rest_idx = all_idx.masked_select(mask).reshape(b, self.nkv, -1)

        lm = landmark_cand.gather(
            2, rest_idx.unsqueeze(-1).expand(-1, -1, -1, self.hd)
        )
        self._k_landmark[layer_idx] = lm
        self._k_landmark_idx[layer_idx] = rest_idx

        self.sparse_start = self.prefill_local + outlier_n * self.chunk_size
        self.sparse_end = self.sparse_start + self.sparse_budget
        self.prefilled = True

        if query_last is not None:
            return self.retrieve(layer_idx, query_last)
        return None

    # ── Decode ───────────────────────────────────────────────────────────

    @torch.no_grad()
    def retrieve(self, layer_idx: int, query: torch.Tensor) -> torch.Tensor:
        """Top-k chunks vs landmarks → position_ids [b, nkv, sparse_budget]."""
        lm = self._k_landmark.get(layer_idx)
        lm_idx = self._k_landmark_idx.get(layer_idx)
        b = query.shape[0]

        if lm is None or lm_idx is None or self.chunks == 0:
            # Short ctx: take last min(sparse_budget, seq_len) positions, pad by repeat
            n = min(self.sparse_budget, max(1, self.seq_len))
            start = max(0, self.seq_len - n)
            base = torch.arange(start, start + n, device=query.device)
            if n < self.sparse_budget:
                pad = self.sparse_budget - n
                base = torch.cat([base, base[-1:].expand(pad)])
            pos = base.view(1, 1, -1).expand(b, self.nkv, -1).contiguous()
            self._last_pos = pos
            return pos

        # query: [b, q, nqh, hd]
        q_len = query.shape[1]
        nqh = query.shape[2]
        g = max(1, nqh // self.nkv)
        qg = query.reshape(b, self.nkv, g, q_len, self.hd)
        scores = torch.einsum("bngqd,bnld->bngql", qg, lm.transpose(2, 3))
        scores = scores / math.sqrt(self.hd)
        scores = torch.softmax(scores.float(), dim=-1).to(query.dtype)
        chunk_attn = scores.sum(dim=3)  # [b, nkv, g, n_lm]
        if g > 1:
            chunk_attn, _ = chunk_attn.max(dim=2)
        else:
            chunk_attn = chunk_attn.squeeze(2)

        k = min(self.select_sets, chunk_attn.shape[-1])
        top = torch.topk(chunk_attn, k=k, dim=-1).indices
        selected = lm_idx.gather(-1, top)

        ar = torch.arange(self.chunk_size, device=query.device).view(1, 1, 1, -1)
        pos = (selected.unsqueeze(-1) * self.chunk_size + ar).reshape(b, self.nkv, -1)
        if pos.shape[-1] < self.sparse_budget:
            pad = self.sparse_budget - pos.shape[-1]
            pos = torch.cat([pos, pos[..., -1:].expand(-1, -1, pad)], dim=-1)
        elif pos.shape[-1] > self.sparse_budget:
            pos = pos[..., : self.sparse_budget]
        self._last_pos = pos
        return pos

    @torch.no_grad()
    def reconstruct(self, layer_idx: int, position_ids: torch.Tensor,
                    rope_fn=None) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct K/V at position_ids via U @ SV.

        Returns (k_sel, v_sel): [b, nkv, sparse_budget, hd]
        """
        U_k = self.U_k.get(layer_idx)
        SV_k = self.SV_k.get(layer_idx)
        U_v = self.U_v.get(layer_idx)
        SV_v = self.SV_v.get(layer_idx)
        if U_k is None or U_v is None or SV_k is None or SV_v is None:
            raise RuntimeError(f"xKV: missing SVD factors for layer {layer_idx}")

        # U: [b, s, r] → [b, nkv, s, r] → gather
        r = U_k.shape[-1]
        idx = position_ids.unsqueeze(-1).expand(-1, -1, -1, r)
        U_exp = U_k.unsqueeze(1).expand(-1, self.nkv, -1, -1)
        U_h = torch.gather(U_exp, 2, idx)  # [b, nkv, B, r]

        # SV: [b, nkv, r, hd]  U_h: [b, nkv, B, r] → [b, nkv, B, hd]
        k_sel = torch.einsum("bnhr,bnrd->bnhd", U_h, SV_k).contiguous()

        r_v = U_v.shape[-1]
        idx_v = position_ids.unsqueeze(-1).expand(-1, -1, -1, r_v)
        U_v_exp = U_v.unsqueeze(1).expand(-1, self.nkv, -1, -1)
        U_vh = torch.gather(U_v_exp, 2, idx_v)
        v_sel = torch.einsum("bnhr,bnrd->bnhd", U_vh, SV_v).contiguous()

        if rope_fn is not None:
            k_sel = rope_fn(k_sel, position_ids)

        return k_sel, v_sel

    # ── Train-time fake SVD ──────────────────────────────────────────────

    @staticmethod
    @torch.no_grad()
    def compress_group_fake(k_flat: torch.Tensor, v_flat: torch.Tensor,
                            rank_k: int, rank_v: int) -> tuple[torch.Tensor, torch.Tensor]:
        return fake_svd(k_flat, rank_k), fake_svd(v_flat, rank_v)

    def stats(self) -> dict:
        return {
            "group_size": self.group_size,
            "rank_k": self.rank_k,
            "rank_v": self.rank_v,
            "sparse_budget": self.sparse_budget,
            "chunk_size": self.chunk_size,
            "seq_len": self.seq_len,
            "chunks": self.chunks,
            "n_layers_svd": len(self.U_k),
        }


def make_default_config() -> XKVConfig:
    """moe-flash defaults: gs=4, rank 128, budget 2048, chunk 8."""
    return XKVConfig(
        group_size=4,
        rank_k=128,
        rank_v=128,
        sparse_budget=2048,
        chunk_size=8,
        local_window=32,
        enabled=True,
    )
