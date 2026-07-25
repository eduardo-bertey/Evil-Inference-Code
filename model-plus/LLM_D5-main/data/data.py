import os
import json
import time
import unicodedata
import threading
import queue
import numpy as np
from tqdm import tqdm
from datasets import load_dataset, interleave_datasets
from transformers import AutoTokenizer
from huggingface_hub import login

# ── Auth via environment variable (never hardcode tokens) ──────────────────────
hf_token = "TOKEN FOR MORE RELIABLE DOWNLOAD"
if hf_token:
    login(token=hf_token)

# ── Load custom tokenizer ──────────────────────────────────────────────────────
enc = AutoTokenizer.from_pretrained("./custom_multilingual_tokenizer", use_fast=True)

bos_id = enc.bos_token_id if enc.bos_token_id is not None else enc.convert_tokens_to_ids("<|bos|>")
eos_id = enc.eos_token_id if enc.eos_token_id is not None else enc.convert_tokens_to_ids("<|endoftext|>")

print(f"bos_id: {bos_id}  eos_id: {eos_id}")
assert bos_id is not None and bos_id >= 0, "bos_id is invalid"
assert eos_id is not None and eos_id >= 0, "eos_id is invalid"


# ─────────────────────────────────────────────────────────────────────────────
# DATASET DISTRIBUTION  — weights MUST sum to exactly 1.00
# ─────────────────────────────────────────────────────────────────────────────
datasets_config = [
    {"path": "HuggingFaceFW/fineweb-edu",  "name": "sample-100BT", "text_col": "text",    "weight": 0.340},
    {"path": "open-web-math/open-web-math", "name": "default",      "text_col": "text",    "weight": 0.100},
    {"path": "bigcode/starcoderdata",       "data_dir": "python",   "text_col": "content", "weight": 0.100},
    {"path": "bigcode/starcoderdata",       "data_dir": "c",        "text_col": "content", "weight": 0.050},
    {"path": "uonlp/CulturaX",              "name": "ru",           "text_col": "text",    "weight": 0.360},
    {"path": "wikimedia/wikipedia",         "name": "20231101.ru",  "text_col": "text",    "weight": 0.040},
    {"path": "uonlp/CulturaX",              "name": "uz",           "text_col": "text",    "weight": 0.006},
    {"path": "wikimedia/wikipedia",         "name": "20231101.uz",  "text_col": "text",    "weight": 0.004},
]

# Sanity-check weights at startup
_total_w = sum(c["weight"] for c in datasets_config)
assert abs(_total_w - 1.0) < 1e-6, f"Weights must sum to 1.0, got {_total_w:.6f}"


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
TOTAL_ROWS_TARGET  = 150_000_000
BATCH_SIZE         = 10_000      # Large batch for fast Rust tokenization
MIN_TEXT_LEN       = 10
dtype              = np.uint16   # safe for vocab ≤ 65,535

data_dir  = "data/"
meta_path = os.path.join(data_dir, "train_multilingual.meta.json")
os.makedirs(data_dir, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def get_file_path() -> str:
    return os.path.join(data_dir, "train.bin")

def load_meta() -> dict:
    """Load persisted progress metadata, or return a fresh state."""
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"📊 Resuming from metadata:")
        print(f"   raw_rows_consumed : {meta['raw_rows_consumed']:,}")
        print(f"   valid_rows_written: {meta['valid_rows_written']:,}")
        print(f"   tokens_written    : {meta['tokens_written']:,}")
        return meta
    return {"raw_rows_consumed": 0, "valid_rows_written": 0, "tokens_written": 0}

def save_meta(meta: dict):
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

def tokenize_batch(texts: list[str]) -> list[int]:
    """Tokenize a batch and wrap every document with <bos>…<eos>."""
    tokenized = enc(
        texts, 
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False
    )
    all_ids: list[int] = []
    for ids in tokenized["input_ids"]:
        all_ids.append(bos_id)
        all_ids.extend(ids)
        all_ids.append(eos_id)
    return all_ids

def write_chunk(f, ids: list[int]) -> int:
    arr = np.array(ids, dtype=dtype)
    f.write(arr.tobytes())
    return len(ids)

def background_batched_fetcher(dataset_iterator, batch_size, buffer_size=3):
    """Fetches from HF and batches locally in a background thread."""
    q = queue.Queue(maxsize=buffer_size)
    
    def _producer():
        try:
            batch_texts = []
            for item in dataset_iterator:
                batch_texts.append(item["text"])
                if len(batch_texts) >= batch_size:
                    q.put({"text": batch_texts})
                    batch_texts = []
            
            # Push any remaining items at the very end of the dataset
            if batch_texts:
                q.put({"text": batch_texts})
        except Exception as e:
            print(f"\n⚠️ Stream interrupted in background worker: {e}")
        finally:
            q.put(None) # Sentinel value to signal completion
            
    t = threading.Thread(target=_producer, daemon=True)
    t.start()
    
    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    meta = load_meta()
    raw_rows_done   = meta["raw_rows_consumed"]
    valid_rows_done = meta["valid_rows_written"]
    tokens_done     = meta["tokens_written"]

    if valid_rows_done >= TOTAL_ROWS_TARGET:
        print("✅ Target already reached. Nothing to do.")
        exit()

    # ── Build dataset streams ────────────────────────────────────────────────
    print(f"\n🚀 Setting up multilingual streams...")
    streams, valid_configs = [], []

    for config in datasets_config:
        label = config.get("name", config.get("data_dir", "default"))
        print(f"🔗 Connecting to {config['path']} ({label})...")
        time.sleep(0.1)
        try:
            ds = load_dataset(
                path=config["path"],
                name=config.get("name"),
                data_dir=config.get("data_dir"),
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            if config["text_col"] != "text":
                ds = ds.rename_column(config["text_col"], "text")
            streams.append(ds.select_columns(["text"]))
            valid_configs.append(config)
        except Exception as e:
            print(f"❌ Skipping {config['path']}: {e}")

    if not streams:
        raise RuntimeError("No datasets loaded — aborting.")

    total_w = sum(c["weight"] for c in valid_configs)
    probabilities = [c["weight"] / total_w for c in valid_configs]

    print("\n🔀 Interleaving streams...")
    mixed_stream = interleave_datasets(
        streams,
        probabilities=probabilities,
        seed=42,
        stopping_strategy="all_exhausted",
    )
    mixed_stream = mixed_stream.shuffle(buffer_size=10_000, seed=42)

    if raw_rows_done > 0:
        print(f"⏭️  Skipping {raw_rows_done:,} raw stream rows to resume safely...")
        mixed_stream = mixed_stream.skip(raw_rows_done)

    # ── CUSTOM BATCHING & PREFETCHING ────────────────────────────────────────
    fast_iterator = background_batched_fetcher(iter(mixed_stream), BATCH_SIZE, buffer_size=3)

    remaining_target = TOTAL_ROWS_TARGET - valid_rows_done
    current_file = open(get_file_path(), "ab")

    print(f"✍️  Writing to {get_file_path()}...")

    try:
        with tqdm(total=TOTAL_ROWS_TARGET, initial=valid_rows_done, desc="Tokenizing", unit="rows") as pbar:
            for batch in fast_iterator:
                if valid_rows_done >= TOTAL_ROWS_TARGET:
                    break

                raw_texts = batch["text"]
                raw_rows_in_batch = len(raw_texts)
                
                # Fast clean and filter
                batch_text = [
                    unicodedata.normalize("NFC", t).strip() 
                    for t in raw_texts 
                    if len(t) >= MIN_TEXT_LEN
                ]
                
                # We always account for how many raw rows we pulled from the stream
                raw_rows_done += raw_rows_in_batch

                if not batch_text:
                    continue

                valid_rows_in_batch = len(batch_text)
                valid_rows_done += valid_rows_in_batch

                # Core performance step: tokenization and flushing
                ids = tokenize_batch(batch_text)
                tokens_done += write_chunk(current_file, ids)
                
                pbar.update(valid_rows_in_batch)

                # Periodically save metadata
                if raw_rows_done % 100_000 == 0:
                     save_meta({
                        "raw_rows_consumed": raw_rows_done,
                        "valid_rows_written": valid_rows_done,
                        "tokens_written": tokens_done,
                    })

    finally:
        current_file.close()
        save_meta({
            "raw_rows_consumed": raw_rows_done,
            "valid_rows_written": valid_rows_done,
            "tokens_written": tokens_done,
        })

    print(f"\n✅ Done!  Valid rows={valid_rows_done:,} (Raw stream checked={raw_rows_done:,})  Tokens={tokens_done:,}")