"""Dataset handling: download Wikipedia ES for tokenizer and training data."""

import os, threading
from datasets import load_dataset

_DIR = os.path.dirname(os.path.abspath(__file__))
WIKI_CONFIG = ("wikimedia/wikipedia", "20231101.es")
TOKENIZER_DATA_PATH = os.path.join(_DIR, "wiki_tokenizer_50mb.txt")
TRAIN_DATA_PATH = os.path.join(_DIR, "wiki_train_data.txt")


def download_wikipedia_50mb(output_path: str = TOKENIZER_DATA_PATH) -> str:
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


class StreamingDataset:
    """Stream training data in blocks via persistent iterator with prefetch."""
    def __init__(self, block_mb: float = 3.0, block_idx: int = 0):
        self.block_mb = block_mb
        self.block_idx = block_idx
        self._path = os.path.join(_DIR, f"wiki_block_{block_idx}.txt")
        self._tokens = None
        self._tokenizer = None
        self._wiki_iter = None
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_error: Exception | None = None

    def _ensure_wiki_iter(self):
        if self._wiki_iter is None:
            ds = load_dataset(*WIKI_CONFIG, split="train", streaming=True)
            self._wiki_iter = iter(ds)

    def download_block(self):
        max_bytes = int(self.block_mb * 1024 * 1024)
        self._ensure_wiki_iter()
        written = 0
        with open(self._path, "w", encoding="utf-8") as f:
            for item in self._wiki_iter:
                text = f"--- {item['title']} ---\n{item['text']}\n\n"
                tam = len(text.encode("utf-8"))
                if written + tam > max_bytes:
                    break
                f.write(text)
                written += tam
        if written < max_bytes:
            print(f"  Wikipedia stream exhausted at block {self.block_idx}, wrapping on next block")

    def _prefetch_worker(self, block_idx: int):
        try:
            self._prefetch_error = None
            path = os.path.join(_DIR, f"wiki_block_{block_idx}.txt")
            if not os.path.exists(path):
                old_block = self.block_idx
                old_path = self._path
                self.block_idx = block_idx
                self._path = path
                self.download_block()
                self.block_idx = old_block
                self._path = old_path
        except Exception as e:
            self._prefetch_error = e

    def _wait_prefetch(self):
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            self._prefetch_thread.join()
            self._prefetch_thread = None
        if self._prefetch_error is not None:
            err = self._prefetch_error
            self._prefetch_error = None
            print(f"  Prefetch failed for block {self.block_idx + 1}: {err}")

    def _start_prefetch(self, block_idx: int):
        self._wait_prefetch()
        self._prefetch_thread = threading.Thread(target=self._prefetch_worker, args=(block_idx,), daemon=True)
        self._prefetch_thread.start()

    def load_tokens(self, tokenizer):
        self._tokenizer = tokenizer
        if not os.path.exists(self._path):
            self.download_block()
        with open(self._path, "r", encoding="utf-8") as f:
            text = f.read()
        self._tokens = tokenizer.encode(text)
        print(f"Loaded {len(self._tokens)} tokens from block {self.block_idx}")
        self._start_prefetch(self.block_idx + 1)

    def next_block(self):
        old_path = os.path.join(_DIR, f"wiki_block_{self.block_idx}.txt")
        if os.path.exists(old_path):
            os.remove(old_path)
        self._tokens = None
        self._wait_prefetch()
        self.block_idx += 1
        self._path = os.path.join(_DIR, f"wiki_block_{self.block_idx}.txt")
        if not os.path.exists(self._path):
            self.download_block()
        with open(self._path, "r", encoding="utf-8") as f:
            text = f.read()
        self._tokens = self._tokenizer.encode(text)
        if len(self._tokens) < 1000:
            os.remove(self._path)
            print(f"  Dataset exhausted at block {self.block_idx}, wrapping to block 0")
            self.block_idx = 0
            self._path = os.path.join(_DIR, f"wiki_block_{self.block_idx}.txt")
            self._wiki_iter = None
            if not os.path.exists(self._path):
                self.download_block()
            with open(self._path, "r", encoding="utf-8") as f:
                text = f.read()
            self._tokens = self._tokenizer.encode(text)
        print(f"  Loaded block {self.block_idx}: {len(self._tokens)} tokens")
        self._start_prefetch(self.block_idx + 1)

    def get_tokens(self):
        if self._tokens is None:
            raise ValueError("Call load_tokens() first")
        return self._tokens
