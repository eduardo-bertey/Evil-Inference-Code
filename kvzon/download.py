"""Descarga tokenizer + checkpoint de laurelia-llm para kvzon.

Baja de ScortexIA/laurelia@laurelia-llm:
  tokenizer.json (~2.3MB) + checkpoint.pt (~652MB)
Salta lo que ya existe en kvzon/.
"""

import os
import sys

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    sys.exit("Falta huggingface_hub: pip install huggingface_hub")

REPO_ID = "ScortexIA/laurelia"
REVISION = "laurelia-llm"
FILES = ("tokenizer.json", "checkpoint.pt")

_KVZON = os.path.dirname(os.path.abspath(__file__))


def main():
    for fn in FILES:
        dest = os.path.join(_KVZON, fn)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            print(f"{fn}: ya existe ({os.path.getsize(dest)} bytes), skip")
            continue
        print(f"Bajando {fn} de {REPO_ID}@{REVISION}...")
        path = hf_hub_download(repo_id=REPO_ID, revision=REVISION, filename=fn)
        import shutil
        shutil.copy2(path, dest)
        print(f"{fn}: OK ({os.path.getsize(dest)} bytes)")


if __name__ == "__main__":
    main()
