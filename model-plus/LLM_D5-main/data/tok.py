import os
import time
from datasets import load_dataset, interleave_datasets
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
from huggingface_hub import login

# ── Auth via environment variable (never hardcode tokens) ──────────────────────
hf_token = "YOUT TOKENS"


def build_tokenizer():
    # ─────────────────────────────────────────────────────────────────
    # 1. Balanced Tokenizer Training Distribution (Sum = 1.00)
    # ─────────────────────────────────────────────────────────────────
    datasets_config = [
        {"path": "HuggingFaceFW/fineweb-edu",  "name": "sample-10BT",    "weight": 0.35},
        {"path": "open-web-math/open-web-math", "name": "default",        "weight": 0.10},
        {"path": "bigcode/starcoderdata",        "data_dir": "python",     "weight": 0.10},
        {"path": "bigcode/starcoderdata",        "data_dir": "c",          "weight": 0.05},
        {"path": "uonlp/CulturaX",              "name": "ru",             "weight": 0.25},
        {"path": "wikimedia/wikipedia",          "name": "20231101.ru",    "weight": 0.05},
        {"path": "uonlp/CulturaX",              "name": "uz",             "weight": 0.08},
        {"path": "wikimedia/wikipedia",          "name": "20231101.uz",    "weight": 0.02},
    ]
    # Verify weights sum to 1.0
    total_weight = sum(c["weight"] for c in datasets_config)
    assert abs(total_weight - 1.0) < 1e-6, f"Weights must sum to 1.0, got {total_weight:.4f}"

    streams = []
    valid_configs = []
    for config in datasets_config:
        label = config.get("name", config.get("data_dir", "default"))
        print(f"🔗 Connecting to {config['path']} ({label})...")
        time.sleep(0.5)

        try:
            ds = load_dataset(
                config["path"],
                name=config.get("name"),
                data_dir=config.get("data_dir"),
                split="train",
                streaming=True,
                trust_remote_code=True,
            )

            # Unify text column to 'text'
            if "text" not in ds.column_names:
                col = "content" if "content" in ds.column_names else ds.column_names[0]
                ds = ds.rename_column(col, "text")

            streams.append(ds.select_columns(["text"]))
            valid_configs.append(config)
        except Exception as e:
            print(f"❌ Failed to load {config['path']}: {e}")

    if not streams:
        raise RuntimeError("No datasets could be loaded. Aborting.")

    # Re-normalise weights in case some datasets failed to load
    total = sum(c["weight"] for c in valid_configs)
    probabilities = [c["weight"] / total for c in valid_configs]

    print("\n🔀 Interleaving streams for tokenizer distribution...")
    mixed_stream = interleave_datasets(
        streams,
        probabilities=probabilities,
        seed=42,
        stopping_strategy="all_exhausted",
    )

    # ─────────────────────────────────────────────────────────────────
    # 2. Document Generator Flow
    # ─────────────────────────────────────────────────────────────────
    TRAIN_DOCS = 2_500_000

    def batch_iterator(batch_size=1000):
        batch = []
        for i, example in enumerate(mixed_stream):
            if i >= TRAIN_DOCS:
                break
            text = example["text"].strip()
            if len(text) > 10:
                batch.append(text)
            if len(batch) == batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    # ─────────────────────────────────────────────────────────────────
    # 3. Initialize and Train the BPE Tokenizer
    # ─────────────────────────────────────────────────────────────────
    print(f"\n🧠 Training custom tokenizer on {TRAIN_DOCS:,} documents...")
    tokenizer = ByteLevelBPETokenizer()

    # Vocab capped at 65,530 → fits safely inside uint16 tensors (max 65,535)
    tokenizer.train_from_iterator(
        batch_iterator(),
        vocab_size=65_530,
        min_frequency=5,
        special_tokens=[
            "<|endoftext|>",
            "<|pad|>",
            "<|bos|>",
            "<|unk|>",
        ],
    )

    # ─────────────────────────────────────────────────────────────────
    # 4. Wrap and Save for HuggingFace Compatibility
    # ─────────────────────────────────────────────────────────────────
    print("\n💾 Saving custom tokenizer...")
    save_dir = "./custom_multilingual_tokenizer"
    os.makedirs(save_dir, exist_ok=True)

    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer._tokenizer,
        bos_token="<|bos|>",
        eos_token="<|endoftext|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
    )

    hf_tokenizer.save_pretrained(save_dir)
    print(f"✅ Tokenizer saved to '{save_dir}'!")


if __name__ == "__main__":
    build_tokenizer()