"""FineWeb (FineWeb2-HQ es) en streaming, por bloques en disco."""

import os

_DIR = os.path.dirname(os.path.abspath(__file__))
BLOCK_MB = 4.0
BLOCK_FILE = lambda i: os.path.join(_DIR, f"k2moe_block_{i}.txt")


def fineweb_iter():
    from datasets import load_dataset
    return iter(load_dataset("epfml/FineWeb2-HQ", "spa_Latn",
                             split="train", streaming=True))


def skip_blocks(it, skip, block_mb):
    max_bytes = int(block_mb * 1024 * 1024)
    for b in range(skip):
        written, saw = 0, False
        for item in it:
            saw = True
            text = item.get("text") if isinstance(item, dict) else str(item)
            tam = len(text.encode("utf-8"))
            if written + tam > max_bytes:
                break
            written += tam
        if not saw:
            print("  FineWeb agotado al saltar bloques")
            return False
        if (b + 1) % 20 == 0 or b + 1 == skip:
            print(f"  skip {b + 1}/{skip} bloques")
    return True


class FineWebBlocks:
    def __init__(self, block_idx=0, block_mb=BLOCK_MB):
        self.block_idx = block_idx
        self.block_mb = block_mb
        self._tokens = None
        self._tokenizer = None
        self._iter = None

    def _path(self):
        return BLOCK_FILE(self.block_idx)

    def _download(self):
        if self._iter is None:
            print("Abriendo stream epfml/FineWeb2-HQ:spa_Latn...")
            self._iter = fineweb_iter()
            if self.block_idx > 0:
                skip_blocks(self._iter, self.block_idx, self.block_mb)
        max_bytes = int(self.block_mb * 1024 * 1024)
        print(f"  Descargando bloque {self.block_idx} (~{self.block_mb}MB)...")
        buf, n = bytearray(), 0
        for item in self._iter:
            text = item.get("text") if isinstance(item, dict) else str(item)
            tam = len(text.encode("utf-8"))
            if len(buf) + tam > max_bytes:
                break
            buf += text.encode("utf-8") + b"\n\n"
            n += 1
        with open(self._path(), "wb") as f:
            f.write(buf)
        print(f"  Escrito {self._path()}: {len(buf)} bytes ({n} items)")

    def load_tokens(self, tokenizer):
        self._tokenizer = tokenizer
        if not os.path.exists(self._path()):
            self._download()
        with open(self._path(), "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        self._tokens = self._tokenizer.encode(text)
        print(f"  Bloque {self.block_idx}: {len(text)} bytes | {len(self._tokens)} tokens")

    def next_block(self):
        old = self._path()
        if os.path.exists(old):
            try:
                os.remove(old)
            except OSError:
                pass
        self._tokens = None
        self.block_idx += 1
        self._download()
        self.load_tokens(self._tokenizer)

    def get_tokens(self):
        return self._tokens
