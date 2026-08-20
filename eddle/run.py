import json
import pickle
import torch
from .architecture import SimpleAttentionNetwork, TransformerConfig, make_causal_mask, make_padding_mask


def load_checkpoint(path, device="cpu"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    config = TransformerConfig(**data["config"])
    model = SimpleAttentionNetwork(config)
    state_dict = data.get("params", data.get("state_dict", None))
    if state_dict is None:
        raise ValueError("Checkpoint missing 'params' or 'state_dict'")
    if isinstance(state_dict, dict):
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    return model, config


def _build_encoder_input(tokenizer, query, tools, max_enc_len=1024):
    tools_sep_id = tokenizer.tools_token_id
    q_toks = tokenizer.encode(query)
    t_toks = tokenizer.encode(tools)
    max_query = max_enc_len - 2
    if len(q_toks) > max_query:
        q_toks = q_toks[:max_query]
    remaining = max_enc_len - len(q_toks) - 1
    t_toks = t_toks[:remaining]
    return q_toks + [tools_sep_id] + t_toks


@torch.no_grad()
def generate(model, tokenizer, query, tools="[]", max_gen_len=512, max_enc_len=1024, device="cpu"):
    model.eval()
    enc_tokens = _build_encoder_input(tokenizer, query, tools, max_enc_len)
    enc_input = torch.tensor([enc_tokens], dtype=torch.long, device=device)

    src_mask = make_padding_mask(enc_input, tokenizer.pad_token_id)
    encoder_out = model.encode(enc_input, src_mask=src_mask)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    dec_buffer = torch.full((1, max_gen_len), pad_id, dtype=torch.long, device=device)
    dec_buffer[0, 0] = eos_id

    tgt_mask = make_causal_mask(max_gen_len, device=device)
    generated = []

    for i in range(max_gen_len - 1):
        logits = model.decode(dec_buffer, encoder_out, self_mask=tgt_mask)
        next_token = logits[0, i].argmax().item()
        if next_token == eos_id:
            break
        generated.append(next_token)
        dec_buffer[0, i + 1] = next_token

    result = tokenizer.decode(generated)
    if result.startswith("<tool_call>"):
        result = result[len("<tool_call>"):]
    return result


@torch.no_grad()
def generate_batch(model, tokenizer, queries, tools_list, max_gen_len=512, max_enc_len=1024, device="cpu"):
    model.eval()
    B = len(queries)
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    enc_token_lists = [_build_encoder_input(tokenizer, q, t, max_enc_len) for q, t in zip(queries, tools_list)]
    max_enc = max(len(toks) for toks in enc_token_lists)
    enc_input = torch.full((B, max_enc), pad_id, dtype=torch.long, device=device)
    for i, toks in enumerate(enc_token_lists):
        enc_input[i, :len(toks)] = toks

    src_mask = make_padding_mask(enc_input, pad_id)
    encoder_out = model.encode(enc_input, src_mask=src_mask)

    dec_buffer = torch.full((B, max_gen_len), pad_id, dtype=torch.long, device=device)
    dec_buffer[:, 0] = eos_id
    tgt_mask = make_causal_mask(max_gen_len, device=device)

    finished = [False] * B
    gen_tokens = [[] for _ in range(B)]

    for pos in range(max_gen_len - 1):
        logits = model.decode(dec_buffer, encoder_out, self_mask=tgt_mask)
        for i in range(B):
            if finished[i]:
                continue
            next_token = logits[i, pos].argmax().item()
            if next_token == eos_id:
                finished[i] = True
                continue
            gen_tokens[i].append(next_token)
            dec_buffer[i, pos + 1] = next_token
        if all(finished):
            break

    results = []
    for i in range(B):
        text = tokenizer.decode(gen_tokens[i])
        if text.startswith("<tool_call>"):
            text = text[len("<tool_call>"):]
        results.append(text)
    return results
