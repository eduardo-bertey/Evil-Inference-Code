"""Dataset handling: download Wikipedia ES and Spanish FineWeb2-HQ training data."""

import os
from typing import Optional
from datasets import load_dataset

_DIR = os.path.dirname(os.path.abspath(__file__))
WIKI_CONFIG = ("wikimedia/wikipedia", "20231101.es")
FINEWEB_CONFIG = ("epfml/FineWeb2-HQ", "spa_Latn")
TOKENIZER_DATA_PATH = os.path.join(_DIR, "wiki_tokenizer_50mb.txt")
TRAIN_DATA_PATH = os.path.join(_DIR, "wiki_train_data.txt")


def _append_mix_content(output_path: str, mix_mb: float, block_idx: int = 0, mix_dataset: Optional[str] = None):
    """Append up to `mix_mb` megabytes of text to `output_path`.

    Behavior:
    - If `mix_dataset` is provided, try to stream from that dataset id.
    - Otherwise, look for a local file `fineweb_block_{block_idx}.txt` in the same dir.
    """
    mix_bytes = int(mix_mb * 1024 * 1024)
    appended = 0

    # Try remote dataset first if provided
    if mix_dataset:
        print(f"Attempting to append {mix_mb}MB from dataset {mix_dataset}...")
        if isinstance(mix_dataset, (tuple, list)):
            ds2 = load_dataset(*mix_dataset, split="train", streaming=True)
        else:
            ds2 = load_dataset(mix_dataset, split="train", streaming=True)
        it2 = iter(ds2)
        # Skip to block-aligned offset
        skip_bytes = block_idx * mix_bytes
        if skip_bytes > 0:
            skipped = 0
            for item in it2:
                text = item.get("text") if isinstance(item, dict) else str(item)
                skipped += len(text.encode("utf-8"))
                if skipped >= skip_bytes:
                    break
        with open(output_path, "a", encoding="utf-8") as f:
            for item in it2:
                text = item.get("text") if isinstance(item, dict) else str(item)
                tam = len(text.encode("utf-8"))
                if appended + tam > mix_bytes:
                    break
                f.write(text)
                f.write("\n\n")
                appended += tam
        print(f"Appended {appended} bytes from dataset {mix_dataset} to {output_path}")
        return appended

    # Fallback: check for a local fineweb block file
    local_path = os.path.join(_DIR, f"fineweb_block_{block_idx}.txt")
    if os.path.exists(local_path):
        print(f"Appending local fineweb block {local_path} up to {mix_mb}MB")
        with open(local_path, "r", encoding="utf-8") as src, open(output_path, "a", encoding="utf-8") as dst:
            while appended < mix_bytes:
                chunk = src.read(min(65536, mix_bytes - appended))
                if not chunk:
                    break
                dst.write(chunk)
                appended += len(chunk.encode("utf-8"))
        print(f"Appended {appended} bytes from {local_path} to {output_path}")
        return appended

    raise FileNotFoundError("No mix source found (mix_dataset not provided and local fineweb_block file missing)")


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


def download_training_block(
    output_path: str = TRAIN_DATA_PATH,
    block_mb: float = 3.0,
    mezcla: bool = True,
    mix_mb: float = 1.0,
    mix_dataset: Optional[tuple[str, str] | str] = None,
) -> str:
    """Download one block of Wikipedia ES for training.

    Optional mixing: if `mezcla` is True and `mix_mb` > 0, the function will
    attempt to append up to `mix_mb` megabytes of additional text from a
    secondary source. The secondary source can be provided via
    `mix_dataset` (a `datasets` id) or as a local file named
    `fineweb_block_{block_idx}.txt` next to this file.
    """
    mix_dataset = mix_dataset if mix_dataset is not None else FINEWEB_CONFIG

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

    # Append Spanish FineWeb only when explicitly requested.
    if mezcla and mix_mb > 0:
        try:
            _append_mix_content(output_path, mix_mb, block_idx=0, mix_dataset=mix_dataset)
        except Exception as e:
            print(f"Mixing skipped: {e}")

    return output_path


class StreamingDataset:
    """Stream training data in 3MB Wikipedia ES blocks, with optional Spanish FineWeb mix."""
    def __init__(
        self,
        block_mb: float = 3.0,
        block_idx: int = 0,
        mezcla: bool = True,
        mix_mb: float = 1.0,
        mix_dataset: Optional[tuple[str, str] | str] = None,
    ):
        self.block_mb = block_mb
        self.block_idx = block_idx
        self._path = os.path.join(_DIR, f"wiki_block_{block_idx}.txt")
        self._tokens = None
        self._tokenizer = None
        # Mixing options
        self.mezcla = mezcla
        self.mix_mb = mix_mb
        self.mix_dataset = mix_dataset if mix_dataset is not None else FINEWEB_CONFIG

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

        # If mixing is enabled, try to append content from secondary source
        if getattr(self, "mezcla", False) and self.mix_mb > 0:
            try:
                _append_mix_content(self._path, self.mix_mb, self.block_idx, mix_dataset=self.mix_dataset)
            except Exception as e:
                print(f"Mixing skipped for block {self.block_idx}: {e}")

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