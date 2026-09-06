"""Chat comparativo laurelia-llm: vanilla vs Q-first (KV modificado).

Vanilla: prefill token-por-token (LLM.generate actual) + cache por capa.
Modificado (Q-first): DOS pasadas por token.
  Pasada 1 (att): Q fija por capa (una vez, no se recalcula) + atencion
          1 vez por capa con K/V del stream actual. No guarda nada.
  Pasada 2 (res): cadena FF con residuales; por capa se GUARDA el vector
          crudo (att+res) como un token mas (cache cruda dim D; K/V se
          proyectan al leer). Capa 1 sin residuo previo.
  Siguiente token: Q, att, FF -> otro vector (cache vector 2, 3, ...).
  50 tokens -> 50 vectores por capa. Atencion siempre dinamica.
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
    """Q fija una vez; att 1 vez por capa con K/V del stream actual (dinamica);
    se guarda (att+res) crudo por capa. Sin recomputos."""

    def __init__(self, model):
        self.m = model
        self.cfg = model.config
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
        """Un token en DOS pasadas (att una vez, sin batch).

        Caches CRUDAS por capa: vectores (att+res) dim D, un token mas por
        step (50 tokens -> 50 vectores). K/V se proyectan al leer.

        Pasada 1: cada capa hace Q+att AL INICIO, sin FF, sin guardar.
        Pasada 2: cadena res+FF por capa (SIN sdpa): att ya calculada +
        residuo ponderado, GUARDAR vector crudo, despues su FF.
        """
        m = self.m
        H = m.blocks[0].attn.num_heads
        G = m.blocks[0].attn.num_kv_groups
        Hd = m.config.dim // H
        dev = tok_id.device
        e = m.embeddings(tok_id)  # (1,1,D)

        # Pasada 1: Q+att por capa al inicio, sin FF, sin guardar.
        A_all = []
        n_a = []
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
            A = attn.o_proj(out.transpose(1, 2).contiguous().view(1, 1, -1))
            n_a.append(float(A.norm()))
            A_all.append(A)

        # Pasada 2: cadena res+FF (SIN sdpa), guardando vector crudo.
        s_prev = None
        new_caches = []
        for li, blk in enumerate(m.blocks):
            A = A_all[li]
            s = e + A if li == 0 else s_prev + A  # UNA sola suma a cache
            new_caches.append(torch.cat([caches[li], s.clone()], dim=1))
            s_prev = s + blk.mlp(blk.ln_2(s))
        logits = m.lm_head(m.norm_f(s_prev))
        if not torch.isfinite(logits).all():
            print(f"  [qfirst] logits no-finitos offset={offset} "
                  f"max|logits|={float(logits.abs().max())} "
                  f"|A|={[f'{v:.1g}' for v in n_a]}")
        return logits, new_caches

    @torch.no_grad()
    def prefill(self, input_ids):
        m = self.m
        H = m.blocks[0].attn.num_heads
        G = m.blocks[0].attn.num_kv_groups
        Hd = m.config.dim // H
        h0 = m.embeddings(input_ids)
        B, P, D = h0.shape

        # Pasada 1: cada capa hace Q+att AL INICIO, sin FF, sin guardar.
        A_all = []
        n_a = []
        for blk in m.blocks:
            attn = blk.attn
            u = blk.ln_1(h0)
            Q = attn.q_proj(u).view(B, P, H, Hd)
            K0 = attn.k_proj(u).view(B, P, G, Hd)
            V0 = attn.v_proj(u).view(B, P, G, Hd)
            Qr, Kr = attn.rope(Q, K0, 0)
            A = attn.o_proj(self._sdpa_prompt(attn, Qr, Kr, V0))
            n_a.append(float(A.norm()))
            A_all.append(A)

        # Pasada 2: cadena res+FF (SIN sdpa). Capa 1 sin residuo previo.
        s_prev = None
        caches = []
        for li, blk in enumerate(m.blocks):
            A = A_all[li]
            s = h0 + A if li == 0 else s_prev + A  # UNA sola suma a cache
            caches.append(s.clone())  # vector crudo (att+res) como tokens
            s_prev = s + blk.mlp(blk.ln_2(s))

        self.diag = {"norm_a": n_a}
        logits = m.lm_head(m.norm_f(s_prev))
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
