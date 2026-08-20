import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_DIR = os.path.dirname(os.path.abspath(__file__))

_HF_DATASET_REPO = "Cactus-Compute/tool-calls"
_HF_TOKENIZED_REPO = "Cactus-Compute/tokenized-tool-calls"
CACHE_DIR = os.path.join(_DIR, ".data_cache")


def load_tool_calls_from_hf(split="train", max_samples=None):
    from datasets import load_dataset
    ds = load_dataset(_HF_DATASET_REPO, split=split, streaming=True)
    samples = []
    for i, ex in enumerate(ds):
        if max_samples and i >= max_samples:
            break
        samples.append(ex)
    return samples


def prepare_tool_call_pairs(samples, tokenizer, max_enc_len=1024, max_dec_len=512):
    eos_id = tokenizer.eos_token_id
    tool_call_id = tokenizer.tool_call_token_id
    tools_sep_id = tokenizer.tools_token_id

    enc_seqs, dec_in_seqs, dec_tgt_seqs = [], [], []
    for ex in samples:
        q_toks = tokenizer.encode(ex["query"])[:max_enc_len - 2]
        t_toks = tokenizer.encode(ex["tools"])
        a_toks = tokenizer.encode(ex["answers"])

        remaining = max_enc_len - len(q_toks) - 1
        enc_seq = q_toks + [tools_sep_id] + t_toks[:remaining]

        if 2 + len(a_toks) + 1 > max_dec_len:
            continue

        enc_seqs.append(enc_seq)
        dec_in_seqs.append([eos_id, tool_call_id] + a_toks)
        dec_tgt_seqs.append([tool_call_id] + a_toks + [eos_id])

    max_enc = max(len(s) for s in enc_seqs)
    max_dec = max(len(s) for s in dec_in_seqs)

    enc_padded = np.full((len(enc_seqs), max_enc), tokenizer.pad_token_id, dtype=np.int32)
    dec_in_padded = np.full((len(dec_in_seqs), max_dec), tokenizer.pad_token_id, dtype=np.int32)
    dec_tgt_padded = np.full((len(dec_tgt_seqs), max_dec), tokenizer.pad_token_id, dtype=np.int32)

    for i, (e, di, dt) in enumerate(zip(enc_seqs, dec_in_seqs, dec_tgt_seqs)):
        enc_padded[i, :len(e)] = e
        dec_in_padded[i, :len(di)] = di
        dec_tgt_padded[i, :len(dt)] = dt

    return enc_padded, dec_in_padded, dec_tgt_padded


class ToolCallDataset(Dataset):
    def __init__(self, enc, dec_in, dec_tgt):
        self.enc = torch.from_numpy(enc).long()
        self.dec_in = torch.from_numpy(dec_in).long()
        self.dec_tgt = torch.from_numpy(dec_tgt).long()

    def __len__(self):
        return len(self.enc)

    def __getitem__(self, idx):
        return self.enc[idx], self.dec_in[idx], self.dec_tgt[idx]


class StreamTextDataset(Dataset):
    def __init__(self, text_blocks, tokenizer, block_size=512):
        self.block_size = block_size
        all_ids = []
        for text in text_blocks:
            all_ids.extend(tokenizer.encode(text))
            all_ids.append(tokenizer.eos_token_id)
        self.ids = all_ids

    def __len__(self):
        return max(0, len(self.ids) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.ids[idx:idx + self.block_size + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y, torch.zeros(1, dtype=torch.long)


def get_wiki_blocks(max_bytes=3_000_000):
    try:
        from datasets import load_dataset
        ds = load_dataset("wikimedia/wikipedia", "20231101.es", split="train", streaming=True)
        blocks = []
        written = 0
        for item in ds:
            text = f"--- {item['title']} ---\n{item['text']}\n\n"
            tam = len(text.encode("utf-8"))
            if written + tam > max_bytes:
                break
            blocks.append(text)
            written += tam
        return blocks
    except Exception as e:
        print(f"Wiki download failed: {e}")
        return []


def get_fineweb_blocks(max_bytes=2_000_000):
    try:
        from datasets import load_dataset
        ds = load_dataset("epfml/FineWeb2-HQ", "spa_Latn", split="train", streaming=True)
        blocks = []
        written = 0
        for item in ds:
            text = item.get("text", "")
            tam = len(text.encode("utf-8"))
            if written + tam > max_bytes:
                break
            blocks.append(text + "\n\n")
            written += tam
        return blocks
    except Exception as e:
        print(f"FineWeb download failed: {e}")
        return []


def get_tweet_blocks(max_bytes=1_000_000):
    try:
        from datasets import load_dataset
        ds = load_dataset("pysentimiento/spanish-tweets", split="train", streaming=True)
        blocks = []
        written = 0
        for item in ds:
            text = item.get("text", "")
            tam = len(text.encode("utf-8"))
            if written + tam > max_bytes:
                break
            blocks.append(text + "\n\n")
            written += tam
        return blocks
    except Exception as e:
        print(f"Tweets download failed: {e}")
        return []


def make_streaming_loader(tokenizer, block_size=512, batch_size=8, mode="evil"):
    if mode == "evil":
        blocks = []
        blocks.extend(get_wiki_blocks())
        blocks.extend(get_fineweb_blocks())
        blocks.extend(get_tweet_blocks())
        if not blocks:
            print("Warning: evil dataset empty, falling back to HF tool-calls")
            samples = load_tool_calls_from_hf("train", max_samples=1000)
            blocks = [ex.get("query", "") + " " + ex.get("answers", "") for ex in samples]
        ds = StreamTextDataset(blocks, tokenizer, block_size)
    else:
        samples = load_tool_calls_from_hf("train", max_samples=5000)
        enc, dec_in, dec_tgt = prepare_tool_call_pairs(samples, tokenizer, block_size, block_size)
        ds = ToolCallDataset(enc, dec_in, dec_tgt)

    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
