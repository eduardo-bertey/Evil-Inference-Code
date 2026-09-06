"""Chat comparativo laurelia-llm: vanilla vs Q-first (KV modificado).

Vanilla: prefill token-por-token (LLM.generate actual) + cache por capa.
Modificado (Q-first, diseno aprobado):
  Fase A: desde embeddings, por capa se proyecta Q y se corre atencion
          con KV provisorias -> vector a^Q por capa (se borra tras usar).
  Fase B: capa por capa: capa 1 usa a_1^Q + Q (exacto); capa 2+ reusa su Q
          ya calculada + residual (Q' = Q + q_proj(dh), lineal = recomputo),
          KV verdaderas del stream real -> a^{Q+res}; se GUARDA el vector
          crudo (att+res) como un token mas por capa (cache cruda dim D;
          K/V se proyectan al leer). Luego FF + residual.
  Decode (atencion-unica, sin re-prefill): por token nuevo, primer Q de
  las 16 capas de una vez (no se recalcula Q: solo evoluciona el residuo)
  + 1 sdpa batch=16; despues cadena FF; se guarda el residuo en cache.

Comandos:
  /temp <f> /topk <i> /topp <f> /rep <f> /ctx <i> /length <i>
  /mode vanilla|qfirst|ambos   /seed <i>   /reset   /exit
Por prompt (modo ambos): texto de cada modo + tok/s prefill/decode +
max|dlogits| por step + primer token divergente (misma seed).
"""

import sys
import os
import time

_KVZON = os.path.dirname(os.path.abspath(__file__))
_LLMDIR = os.path.join(os.path.dirname(_KVZON), "laurelia-llm")
sys.path.insert(0, _KVZON)
sys.path.insert(0, _LLMDIR)

import torch
import torch.nn.functional as F
from model import LLM, Config, repeat_kv
from tokenizers import Tokenizer


# ─── Q-first prefill ─────────────────────────────────────────────────────────
class QFirst:
    """Prefill en dos fases. Solo prefill: el decode usa forward_with_cache."""

    def __init__(self, model):
        self.m = model
        self.cfg = model.config
        self.a_q = []          # vectores a^Q (Fase A); se vacian en Fase B
        self.q_saved = []      # Q pre-rope por capa (Fase A)
        self.diag = {}         # normas por capa para tests

    def _sdpa_prompt(self, attn, Qr, Kr, V):
        H, G = attn.num_heads, attn.num_kv_groups
        B, P, D = Qr.shape[0], Qr.shape[1], self.cfg.dim
        q = Qr.transpose(1, 2)
        k = repeat_kv(Kr, H, G).transpose(1, 2)
        v = repeat_kv(V, H, G).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return out.transpose(1, 2).contiguous().view(B, P, D)

    @torch.no_grad()
    def decode_step(self, tok_id, caches, offset):
        """Un token: atencion POR CAPA (sin batch) con la primera Q.

        Caches CRUDAS por capa: vectores (att+res) dim D, un token mas por
        step. K/V se proyectan al leer (K con rope offset 0).

        1) Entry stream e. Por capa (igual que Attention.forward_with_cache
           del vanilla): Q + K/V nuevos, atencion sobre cache-proyectada +
           posicion nueva. Este pase NO guarda nada.
        2) Cadena por capa: su att + residuo; GUARDAR el vector crudo
           como un token mas; despues su FF.
        """
        m = self.m
        H = m.blocks[0].attn.num_heads
        G = m.blocks[0].attn.num_kv_groups
        Hd = m.config.dim // H
        dev = tok_id.device
        e = m.embeddings(tok_id)  # (1,1,D)

        # 1) Primera Q + atencion por capa (sin batch). No guarda nada.
        A_all = []
        for li, blk in enumerate(m.blocks):
            attn = blk.attn
            u = blk.ln_1(e)
            Q = attn.q_proj(u).view(1, 1, H, Hd)
            K_new = attn.k_proj(u).view(1, 1, G, Hd)
            V_new = attn.v_proj(u).view(1, 1, G, Hd)
            Qr, _ = attn.rope(Q, torch.zeros(1, 1, G, Hd, device=dev), offset)
            _, Kr_new = attn.rope(Q * 0, K_new, offset)
            raw = caches[li]                              # (1,O,D) cruda
            O = raw.shape[1]
            _, Kr_cache = attn.rope(
                torch.zeros(1, O, H, Hd, device=dev),
                attn.k_proj(raw).view(1, O, G, Hd), 0)
            V_cache = attn.v_proj(raw).view(1, O, G, Hd)
            K_full = torch.cat([Kr_cache, Kr_new], dim=1)  # (1,O+1,G,Hd)
            V_full = torch.cat([V_cache, V_new], dim=1)
            q = Qr.transpose(1, 2)
            k = repeat_kv(K_full, H, G).transpose(1, 2)
            v = repeat_kv(V_full, H, G).transpose(1, 2)
            out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            A_all.append(attn.o_proj(out.transpose(1, 2).contiguous().view(1, 1, -1)))

        # 2) Cadena por capa: capa 1 arranca en A_1 (SIN residuo: no hay
        # FF previo); resto: su att + residuo anterior. GUARDAR ese vector.
        s = None
        new_caches = []
        for li, blk in enumerate(m.blocks):
            A = A_all[li]
            s = A if li == 0 else s + A
            new_caches.append(torch.cat([caches[li], s.clone()], dim=1))
            s = s + blk.mlp(blk.ln_2(s))
        logits = m.lm_head(m.norm_f(s))
        if not torch.isfinite(logits).all():
            bad_a = [li for li, a in enumerate(A_all) if not torch.isfinite(a).all()]
            bad_c = [li for li, c in enumerate(new_caches) if not torch.isfinite(c).all()]
            print(f"  [qfirst] logits no-finitos offset={offset} "
                  f"max|logits|={float(logits.abs().max())} A_malas={bad_a} C_malas={bad_c}")
        return logits, new_caches

    @torch.no_grad()
    def prefill(self, input_ids):
        m = self.m
        H = m.blocks[0].attn.num_heads
        G = m.blocks[0].attn.num_kv_groups
        Hd = m.config.dim // H
        h0 = m.embeddings(input_ids)
        B, P, D = h0.shape

        # ── Fase A: Q/K/V sobre todas las capas desde embeddings ──
        # (UNICA vez que se calcula KV del prompt; Fase B la reusa)
        self.q_saved, self.a_q, self.k_saved, self.v_saved = [], [], [], []
        for blk in m.blocks:
            attn = blk.attn
            u = blk.ln_1(h0)
            Q = attn.q_proj(u).view(B, P, H, Hd)          # pre-rope
            K0 = attn.k_proj(u).view(B, P, G, Hd)
            V0 = attn.v_proj(u).view(B, P, G, Hd)
            Qr, Kr = attn.rope(Q, K0, 0)
            A = attn.o_proj(self._sdpa_prompt(attn, Qr, Kr, V0))
            self.q_saved.append(Q)
            self.a_q.append(A)
            self.k_saved.append(Kr)
            self.v_saved.append(V0)

        # ── Fase B: capa 1 arranca en A_1 (SIN residuo: no hay FF previo);
        # resto: su att + residuo anterior. Se GUARDA ese vector.
        s = None
        h_prev = None
        caches = []
        n_q, n_qr = [], []
        for li, blk in enumerate(m.blocks):
            attn = blk.attn
            if li == 0:
                A = self.a_q[li]
            else:
                # Q ya calculada + residual; KV UNA sola vez: reuso Fase A
                uh = blk.ln_1(h_prev)
                u0 = blk.ln_1(h0)
                dQ = attn.q_proj(uh - u0).view(B, P, H, Hd)
                Qp = self.q_saved[li] + dQ
                Qr, _ = attn.rope(Qp, torch.zeros_like(self.v_saved[li]), 0)
                A = attn.o_proj(self._sdpa_prompt(attn, Qr, self.k_saved[li], self.v_saved[li]))
            n_q.append(float(self.a_q[li].norm()))
            n_qr.append(float(A.norm()))
            self.a_q[li] = None  # vector Q-only usado/borrado
            s = A if li == 0 else s + A
            caches.append(s.clone())  # vector crudo (att+res) como tokens
            s = s + blk.mlp(blk.ln_2(s))
            h_prev = s

        self.a_q = []  # todos los vectores Q-only borrados
        self.q_saved = []  # las Q crudas nunca van a cache: se tiran
        self.k_saved = []  # KV provisorias usadas una vez: se tiran
        self.v_saved = []
        self.diag = {"norm_a_q": n_q, "norm_a_qres": n_qr}
        logits = m.lm_head(m.norm_f(s))
        if not torch.isfinite(logits).all():
            print(f"  [qfirst] prefill no-finito max|logits|={float(logits.abs().max())}")
        return logits, caches


# ─── Sampling (copia exacta de LLM.generate) ─────────────────────────────────
def _sample_next(logits_last, temperature, top_k, top_p):
    logits_last = logits_last / temperature
    return logits_last


def _apply_rep_penalty(logits_last, input_ids, repetition_penalty):
    if repetition_penalty != 1.0:
        for tok in input_ids[0].unique():
            if logits_last[0, tok] > 0:
                logits_last[0, tok] /= repetition_penalty
            else:
                logits_last[0, tok] *= repetition_penalty
    return logits_last


def _apply_topk_topp(logits_last, top_k, top_p):
    if top_k > 0:
        v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
        logits_last[logits_last < v[:, [-1]]] = float("-inf")
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits_last, descending=True)
        cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative - torch.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[mask] = float("-inf")
        logits_last.scatter_(1, sorted_idx, sorted_logits)
    return logits_last


@torch.no_grad()
def _decode(model, input_ids, caches, init_logits, prompt_len, max_new, temperature,
            top_k, top_p, repetition_penalty, eos_token_id):
    """Decode identico al de LLM.generate; devuelve (ids, logits_por_step, tps)."""
    step_logits = []
    t0 = time.perf_counter()
    logits = init_logits
    for gen_i in range(max_new):
        logits_last = _sample_next(logits[:, -1, :].clone(), temperature, top_k, top_p)
        logits_last = _apply_rep_penalty(logits_last, input_ids, repetition_penalty)
        logits_last = _apply_topk_topp(logits_last, top_k, top_p)
        probs = torch.softmax(logits_last, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        if eos_token_id is not None and next_tok.item() == eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        logits, caches = model.forward_with_cache(next_tok, prompt_len + gen_i, caches)
        step_logits.append(logits[:, -1, :].float().cpu())
    dt = time.perf_counter() - t0
    n_gen = len(step_logits)
    return input_ids, step_logits, (n_gen / dt if dt > 0 else 0.0)


@torch.no_grad()
def generate_vanilla(model, input_ids, max_new, temperature, top_k, top_p,
                     repetition_penalty, eos_token_id):
    """Replica LLM.generate capturando logits y tiempos."""
    caches = None
    prompt_len = input_ids.shape[1]
    t0 = time.perf_counter()
    for i in range(prompt_len):
        logits, caches = model.forward_with_cache(input_ids[:, i:i + 1], i, caches)
    prefill_tps = prompt_len / max(time.perf_counter() - t0, 1e-9)
    prefill_logits = logits.float().cpu()
    out_ids, step_logits, decode_tps = _decode(
        model, input_ids, caches, logits, prompt_len, max_new, temperature,
        top_k, top_p, repetition_penalty, eos_token_id)
    return out_ids, prefill_logits, step_logits, prefill_tps, decode_tps


@torch.no_grad()
def generate_qfirst(qf, input_ids, max_new, temperature, top_k, top_p,
                    repetition_penalty, eos_token_id):
    """Q-first FIJO: 1 prefill Q-first + N decode_step con atencion-unica.
    Sin re-prefill por token: 50 tokens = 1 prefill + 50 steps baratos."""
    prompt_len = input_ids.shape[1]
    t0 = time.perf_counter()
    logits, caches = qf.prefill(input_ids)
    prefill_tps = prompt_len / max(time.perf_counter() - t0, 1e-9)
    prefill_logits = logits.float().cpu()
    step_logits = []
    t1 = time.perf_counter()
    offset = prompt_len
    for gen_i in range(max_new):
        logits_last = _sample_next(logits[:, -1, :].clone(), temperature, top_k, top_p)
        logits_last = _apply_rep_penalty(logits_last, input_ids, repetition_penalty)
        logits_last = _apply_topk_topp(logits_last, top_k, top_p)
        probs = torch.softmax(logits_last, dim=-1)
        next_tok = torch.multinomial(probs, num_samples=1)
        if eos_token_id is not None and next_tok.item() == eos_token_id:
            break
        input_ids = torch.cat([input_ids, next_tok], dim=1)
        logits, caches = qf.decode_step(next_tok, caches, offset)
        offset += 1
        step_logits.append(logits[:, -1, :].float().cpu())
    dt = time.perf_counter() - t1
    n_gen = len(step_logits)
    return input_ids, prefill_logits, step_logits, prefill_tps, (n_gen / dt if dt > 0 else 0.0)


def compare_steps(logits_a, logits_b):
    """max|dlogits| global + primer step divergente (argmax distinto)."""
    maxd = 0.0
    first_div = None
    for i, (a, b) in enumerate(zip(logits_a, logits_b)):
        d = float((a - b).abs().max())
        maxd = max(maxd, d)
        if first_div is None and int(a.argmax()) != int(b.argmax()):
            first_div = i
    return maxd, first_div


# ─── Chat ────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = Config()

    tok_path = os.path.join(_KVZON, "tokenizer.json")
    ckpt_path = os.path.join(_KVZON, "checkpoint.pt")
    if not os.path.exists(tok_path) or not os.path.exists(ckpt_path):
        print("Faltan pesos en kvzon/, descargando automatico...")
        import download
        download.main()
    tok = Tokenizer.from_file(tok_path)
    config.emb_num = tok.get_vocab_size()
    eos_id = tok.token_to_id("eos_token")
    model = LLM(config).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt["model"].pop("head.emb_weight", None)
    model.load_state_dict(ckpt["model"], strict=False)
    print(f"Loaded checkpoint step {ckpt.get('step', 0)}")
    del ckpt
    model.eval()

    qf = QFirst(model)

    temperature, top_k, top_p = 0.7, 40, 0.9
    repetition_penalty, max_new, max_ctx = 1.2, 200, 1024
    mode, seed = "ambos", 1234

    print("Chat vanilla vs Q-first (Ctrl+C to exit)")
    print("  /temp /topk /topp /rep /ctx /length /mode vanilla|qfirst|ambos /seed /reset /exit")

    while True:
        try:
            prompt = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue
        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0].lower()
            val = parts[1] if len(parts) > 1 else None
            if cmd == "/exit":
                break
            elif cmd == "/temp" and val:
                temperature = float(val)
            elif cmd == "/topk" and val:
                top_k = int(val)
            elif cmd == "/topp" and val:
                top_p = float(val)
            elif cmd == "/rep" and val:
                repetition_penalty = float(val)
            elif cmd == "/ctx" and val:
                max_ctx = int(val)
            elif cmd in ("/length", "/len") and val:
                max_new = int(val)
            elif cmd == "/mode" and val and val in ("vanilla", "qfirst", "ambos"):
                mode = val
                print(f"  mode = {mode}")
            elif cmd == "/seed" and val:
                seed = int(val)
                print(f"  seed = {seed}")
            elif cmd == "/reset":
                print("  (sin estado entre prompts; nada que limpiar)")
            else:
                print(f"  Unknown: {cmd}")
            continue

        ids = tok.encode(prompt).ids[-max_ctx:]
        results = {}
        if mode in ("vanilla", "ambos"):
            torch.manual_seed(seed)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            t = time.perf_counter()
            out, pre_logits, step_logits, pre_tps, dec_tps = generate_vanilla(
                model, x, max_new, temperature, top_k, top_p,
                repetition_penalty, eos_id)
            results["vanilla"] = (out, pre_logits, step_logits, pre_tps, dec_tps,
                                  time.perf_counter() - t)
        if mode in ("qfirst", "ambos"):
            torch.manual_seed(seed)
            x = torch.tensor([ids], dtype=torch.long, device=device)
            t = time.perf_counter()
            out, pre_logits, step_logits, pre_tps, dec_tps = generate_qfirst(
                qf, x, max_new, temperature, top_k, top_p,
                repetition_penalty, eos_id)
            results["qfirst"] = (out, pre_logits, step_logits, pre_tps, dec_tps,
                                 time.perf_counter() - t)

        for name, (out, pre_logits, step_logits, pre_tps, dec_tps, total) in results.items():
            txt = tok.decode(out[0].tolist(), skip_special_tokens=False)
            print(f"--- {name} (prefill {pre_tps:.0f} t/s, decode {dec_tps:.0f} t/s) ---")
            print(txt)

        if mode == "ambos":
            (o1, p1, s1, _, _, _), (o2, p2, s2, _, _, _) = \
                results["vanilla"], results["qfirst"]
            # Vanilla prefill = solo ultimo token (1,1,V); Q-first = prompt
            # completo (1,P,V): comparar ultima posicion contra ultima.
            dp = float((p1[:, -1, :] - p2[:, -1, :]).abs().max())
            maxd, first_div = compare_steps(s1, s2)
            print(f"prefill max|dlogits|={dp:.3g} | decode max|dlogits|={maxd:.3g} | "
                  f"primer token divergente: {first_div}")


if __name__ == "__main__":
    main()
