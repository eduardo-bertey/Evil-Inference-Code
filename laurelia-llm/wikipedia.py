"""Wikipedia ES: descarga del corpus 50MB para el tokenizer y lectura de bloques de entrenamiento."""

import os
from typing import Iterator

from datasets import load_dataset

_DIR = os.path.dirname(os.path.abspath(__file__))
WIKI_CONFIG = ("wikimedia/wikipedia", "20231101.es")
TOKENIZER_DATA_PATH = os.path.join(_DIR, "wiki_tokenizer_50mb.txt")


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


def new_wiki_iter() -> Iterator:
    ds = load_dataset(*WIKI_CONFIG, split="train", streaming=True)
    return iter(ds)


def skip_blocks(iterator, skip_blocks: int, block_mb: float):
    """Salta skip_blocks bloques de wiki y imprime progreso, como dataset.py."""
    max_bytes = int(block_mb * 1024 * 1024)
    for b in range(skip_blocks):
        written = 0
        saw_item = False
        for item in iterator:
            saw_item = True
            text = f"--- {item['title']} ---\n{item['text']}\n\n"
            tam = len(text.encode("utf-8"))
            if tam > max_bytes:
                print(f"  Skipping huge Wikipedia article of {tam} bytes while skipping blocks")
                continue
            if written + tam > max_bytes:
                break
            written += tam
        if not saw_item:
            print(f"  Wikipedia stream exhausted while skipping blocks")
            return 0, True
        if (b + 1) % 20 == 0 or b + 1 == skip_blocks:
            print(f"  wiki skip {b + 1}/{skip_blocks} bloques (~{(b + 1) * block_mb:.0f}MB descargados)")
    return None, False
