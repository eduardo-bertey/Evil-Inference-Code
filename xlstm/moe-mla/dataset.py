"""Dataset handling: download Wikipedia ES for tokenizer and training data."""

import os
from datasets import load_dataset

_DIR = os.path.dirname(os.path.abspath(__file__))
WIKI_CONFIG = ("wikimedia/wikipedia", "20231101.es")
TOKENIZER_DATA_PATH = os.path.join(_DIR, "wiki_tokenizer_50mb.txt")
TRAIN_DATA_PATH = os.path.join(_DIR, "wiki_train_data.txt")


def download_wikipedia_50mb(output_path: str = TOKENIZER_DATA_PATH) -> str:
    """Download first 50MB of Wikipedia ES for tokenizer training."""
    if os.path.exists(output_path) and os.path.getsize(output_path) >= 50_000_000:
        print(f"Tokenizer data already at {output_path} ({os.path.getsize(output_path)} bytes)")
        return output_path

    print("Downloading 50MB Wikipedia ES for tokenizer...")
    ds = load_dataset(*WIKI_CONFIG, split="train", streaming=True)
    with open(output_path, "w", encoding="utf-8") as f:
        written = 0
        for item in ds:
            text = f"--- {item['title']} ---\n{item['text']}\n\n"
            tam = len(text.encode("utf-8"))
            if written + tam > 50_000_000:
                break
            f.write(text)
            written += tam
    print(f"Written {written} bytes to {output_path}")
    return output_path


def download_training_block(output_path: str = TRAIN_DATA_PATH, block_mb: float = 3.0) -> str:
    """Download one 3MB block of Wikipedia ES for training."""
    if os.path.exists(output_path) and os.path.getsize(output_path) >= int(block_mb * 1024 * 1024):
        print(f"Training data already at {output_path} ({os.path.getsize(output_path)} bytes)")
        return output_path

    max_bytes = int(block_mb * 1024 * 1024)
    print(f"Downloading {block_mb}MB Wikipedia ES block...")
    ds = load_dataset(*WIKI_CONFIG, split="train", streaming=True)
    with open(output_path, "w", encoding="utf-8") as f:
        written = 0
        for item in ds:
            text = f"--- {item['title']} ---\n{item['text']}\n\n"
            tam = len(text.encode("utf-8"))
            if written + tam > max_bytes:
                break
            f.write(text)
            written += tam
    print(f"Written {written} bytes to {output_path}")
    return output_path


class StreamingDataset:
    """Stream training data in 3MB blocks, cycling through Wikipedia."""
    def __init__(self, block_mb: float = 3.0, block_idx: int = 0):
        self.block_mb = block_mb
        self.block_idx = block_idx
        self._path = os.path.join(_DIR, f"wiki_block_{block_idx}.txt")
        self._tokens = None
        self._tokenizer = None

    def load_tokens(self, tokenizer):
        self._tokenizer = tokenizer
        self.download_block()
        with open(self._path, "r", encoding="utf-8") as f:
            text = f.read()
        self._tokens = tokenizer.encode(text)
        print(f"Loaded {len(self._tokens)} tokens from block {self.block_idx}")

    def download_block(self):
        max_bytes = int(self.block_mb * 1024 * 1024)
        ds = load_dataset(*WIKI_CONFIG, split="train", streaming=True)
        it = iter(ds)
        # Skip bytes: consume full articles from the SAME iterator
        skip_bytes = self.block_idx * max_bytes
        if skip_bytes > 0:
            skipped = 0
            for item in it:
                text = f"--- {item['title']} ---\n{item['text']}\n\n"
                skipped += len(text.encode("utf-8"))
                if skipped >= skip_bytes:
                    break
        with open(self._path, "w", encoding="utf-8") as f:
            written = 0
            for item in it:
                text = f"--- {item['title']} ---\n{item['text']}\n\n"
                tam = len(text.encode("utf-8"))
                if written + tam > max_bytes:
                    break
                f.write(text)
                written += tam

    def _download_and_load(self):
        """Download block at self.block_idx and load tokens."""
        self._path = os.path.join(_DIR, f"wiki_block_{self.block_idx}.txt")
        self.download_block()
        with open(self._path, "r", encoding="utf-8") as f:
            text = f.read()
        self._tokens = self._tokenizer.encode(text)

    def next_block(self):
        # Delete old block file to free disk
        old_path = os.path.join(_DIR, f"wiki_block_{self.block_idx}.txt")
        if os.path.exists(old_path):
            os.remove(old_path)
        # Drop old tokens before loading new ones
        self._tokens = None
        self.block_idx += 1
        self._download_and_load()
        # If block is too small, dataset is exhausted — wrap to 0
        if len(self._tokens) < 1000:
            os.remove(self._path)
            print(f"  Dataset exhausted at block {self.block_idx}, wrapping to block 0")
            self.block_idx = 0
            self._download_and_load()
        print(f"  Loaded block {self.block_idx}: {len(self._tokens)} tokens")

    def get_tokens(self):
        if self._tokens is None:
            raise ValueError("Call load_tokens() first")
        return self._tokens