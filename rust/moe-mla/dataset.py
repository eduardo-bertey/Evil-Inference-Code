"""Dataset handling: download Wikipedia ES and Spanish FineWeb2-HQ training data."""

import os, threading
from typing import Optional
from datasets import load_dataset

_DIR = os.path.dirname(os.path.abspath(__file__))
WIKI_CONFIG = ("wikimedia/wikipedia", "20231101.es")
FINEWEB_CONFIG = ("epfml/FineWeb2-HQ", "spa_Latn")
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
    """Stream training data in blocks via persistent iterators.
    
    Both Wikipedia and FineWeb2-HQ use persistent load_dataset(streaming=True)
    iterators — they are created once and kept across blocks. No skip, no recreate.
    Prefetch thread downloads the next block while training runs on current block.
    """
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
        self.mezcla = mezcla
        self.mix_mb = mix_mb
        self.mix_dataset = mix_dataset if mix_dataset is not None else FINEWEB_CONFIG
        # Persistent streaming iterators (created once, live forever)
        self._wiki_iter = None
        self._wiki_block_idx = 0
        self._fineweb_iter = None
        self._fineweb_block_idx = 0
        # Prefetch
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_error: Exception | None = None

    def _ensure_wiki_iter(self):
        if self._wiki_iter is None:
            ds = load_dataset(*WIKI_CONFIG, split="train", streaming=True)
            self._wiki_iter = iter(ds)

    def _ensure_fineweb_iter(self):
        if self._fineweb_iter is None and self.mix_dataset is not None:
            if isinstance(self.mix_dataset, (tuple, list)):
                ds = load_dataset(*self.mix_dataset, split="train", streaming=True)
            else:
                ds = load_dataset(self.mix_dataset, split="train", streaming=True)
            self._fineweb_iter = iter(ds)

    def _read_from_fineweb_iter(self, max_bytes: int):
        self._ensure_fineweb_iter()
        if self._fineweb_iter is None:
            return [], 0
        texts = []
        appended = 0
        for item in self._fineweb_iter:
            text = item.get("text") if isinstance(item, dict) else str(item)
            tam = len(text.encode("utf-8"))
            if appended + tam > max_bytes:
                break
            texts.append(text)
            appended += tam
        if appended == 0:
            if self._fineweb_iter is not None:
                print(f"  FineWeb2-HQ stream exhausted at block {self._fineweb_block_idx}, wrapping to block 0")
            self._fineweb_iter = None
            self._fineweb_block_idx = 0
            self._ensure_fineweb_iter()
            if self._fineweb_iter is None:
                return [], 0
            return self._read_from_fineweb_iter(max_bytes)
        self._fineweb_block_idx += 1
        return texts, appended

    def _new_wiki_iter(self):
        ds = load_dataset(*WIKI_CONFIG, split="train", streaming=True)
        return iter(ds)

    def _download_block_from_iterator(self, iterator, skip_blocks: int, path: str) -> int:
        max_bytes = int(self.block_mb * 1024 * 1024)
        for _ in range(skip_blocks):
            written = 0
            for item in iterator:
                text = f"--- {item['title']} ---\n{item['text']}\n\n"
                tam = len(text.encode("utf-8"))
                if written + tam > max_bytes:
                    break
                written += tam
            if written == 0:
                print(f"  Wikipedia stream exhausted while skipping blocks")
                break

        written = 0
        with open(path, "w", encoding="utf-8") as f:
            for item in iterator:
                text = f"--- {item['title']} ---\n{item['text']}\n\n"
                tam = len(text.encode("utf-8"))
                if written + tam > max_bytes:
                    break
                f.write(text)
                written += tam
        return written

    def download_block(self, mix_path=None, iterator=None):
        max_bytes = int(self.block_mb * 1024 * 1024)
        if iterator is None:
            self._ensure_wiki_iter()
            if self._wiki_iter is None or self._wiki_block_idx > self.block_idx:
                self._wiki_iter = self._new_wiki_iter()
                self._wiki_block_idx = 0
            skip_blocks = max(0, self.block_idx - self._wiki_block_idx)
            written = self._download_block_from_iterator(self._wiki_iter, skip_blocks, self._path)
            self._wiki_block_idx = self.block_idx + 1
        else:
            written = self._download_block_from_iterator(iterator, self.block_idx, self._path)

        if written < max_bytes:
            print(f"  Wikipedia stream exhausted while downloading block {self.block_idx}, wrapping on next block")

        if getattr(self, "mezcla", False) and self.mix_mb > 0:
            mix_bytes = int(self.mix_mb * 1024 * 1024)
            try:
                self._ensure_fineweb_iter()
                texts, appended = self._read_from_fineweb_iter(mix_bytes)
                if texts:
                    out_path = mix_path or self._path
                    with open(out_path, "a", encoding="utf-8") as f:
                        for t in texts:
                            f.write(t)
                            f.write("\n\n")
                    print(f"  Appended {appended} bytes from FineWeb2-HQ for block {self._fineweb_block_idx}")
            except Exception as e:
                print(f"  FineWeb mixing skipped: {e}")

    def _prefetch_worker(self, block_idx: int):
        old_block = self.block_idx
        old_path = self._path
        try:
            self._prefetch_error = None
            path = os.path.join(_DIR, f"wiki_block_{block_idx}.txt")
            if not os.path.exists(path):
                self.block_idx = block_idx
                self._path = path
                self.download_block(iterator=self._new_wiki_iter())
        except Exception as e:
            self._prefetch_error = e
        finally:
            self.block_idx = old_block
            self._path = old_path

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

    def _load_tokens_from_file(self):
        with open(self._path, "r", encoding="utf-8") as f:
            text = f.read()
        self._tokens = self._tokenizer.encode(text)

    def load_tokens(self, tokenizer):
        self._tokenizer = tokenizer
        if not os.path.exists(self._path):
            self.download_block()
        self._load_tokens_from_file()
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
        self._load_tokens_from_file()
        if len(self._tokens) < 1000:
            os.remove(self._path)
            print(f"  Dataset exhausted at block {self.block_idx}, wrapping to block 0")
            self.block_idx = 0
            self._path = os.path.join(_DIR, f"wiki_block_{self.block_idx}.txt")
            self._wiki_iter = None
            if not os.path.exists(self._path):
                self.download_block()
            self._load_tokens_from_file()
        print(f"  Loaded block {self.block_idx}: {len(self._tokens)} tokens")
        self._start_prefetch(self.block_idx + 1)

    def get_tokens(self):
        if self._tokens is None:
            raise ValueError("Call load_tokens() first")
        return self._tokens
