"""data.py: arma bloques ES de 64MB y los sube a data-fine-es.

Bloque N (data.{N}.txt):
  [~54.9MB FineWeb2-HQ spa_Latn] + [~9.1MB tweets] (sin alpaca)
Proporcion FineWeb 85.81% / Tweets 14.19% segun peso de cantera (500GB / 82.65GB).
Ambas fuentes se agotan juntas (~9322 bloques): mezcla balanceada.

Reutiliza dataset.py de ESTA carpeta (copia autocontenida del de laurelia-plus:
StreamingDataset con iteradores persistentes, offsets absolutos, prefetch)
+ huggingface.py de esta carpeta.
Paralelo: mientras se SUBE el bloque N, ya se DESCARGA el bloque N+1.

Uso: editar CONFIG arriba y correr:
  !python data.py

El token se pide como en train de laurelia (HF_TOKEN o prompt).
Sin torch: texto puro, corre en CPU.
"""

import concurrent.futures
import os
import re
import sys
import time

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from dataset import StreamingDataset, FINEWEB_CONFIG, TWEETS_CONFIG  # noqa: E402
from huggingface import HFDataManager  # noqa: E402

# ============ CONFIG (editar aca, sin argumentos) ============
REPO = "data-fine-es"   # repo dataset destino (namespace = tu usuario)
BLOQUE_INICIAL = 1      # desde que bloque
COUNT = 0               # cuantos bloques seguidos (0 = todos hasta MAX_BLOQUE)
MAX_BLOQUE = 9000       # tope: al llegar se detiene (evita wrap/repeticion; agote ~9322)
MIN_CORPUS_MB = 32.0    # bajo esto el bloque es basura: reintenta 1 vez, si no, para sin subir
SUBIR = True            # False = solo local, no sube
BAJAR = 0               # >0 = baja data.BAJAR.txt y sale (ignora lo demas)
# =============================================================


# 64 MiB segun peso de cantera (fine 500GB / tuit 82.65GB).
MIXES = [
    (FINEWEB_CONFIG, 54.9, "fine"),
    (TWEETS_CONFIG, 9.1, "tuit"),
]

_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_MENCION = re.compile(r"@\w+")
_RE_HASHTAG = re.compile(r"#\w+")
_RE_ESPACIOS = re.compile(r"\s+")


def limpiar_tuit(t):
    """Saca URLs, @menciones y #hashtags; colapsa espacios."""
    t = _RE_URL.sub("", t)
    t = _RE_MENCION.sub("", t)
    t = _RE_HASHTAG.sub("", t)
    return _RE_ESPACIOS.sub(" ", t).strip()


class BlockDataset(StreamingDataset):
    """StreamingDataset de laurelia que registra bytes por fuente para porcentajes."""

    def __init__(self, block_idx: int):
        super().__init__(block_mb=0.0, block_idx=block_idx, mezcla=True, mixes=MIXES)
        self.last_bytes: dict[str, int] = {}

    def _append_mix_maybe(self, mix_path=None):
        self.last_bytes = {}
        if not getattr(self, "mezcla", False) or not self.mixes:
            return
        for _, mb, label in self.mixes:
            if mb <= 0:
                continue
            mix_bytes = int(mb * 1024 * 1024)
            print(f"  Descargando {label} (bloque {self.block_idx}, {mb}MB)...")
            # Nunca None: espera y reintenta siempre hasta traer el cacho.
            intento = 0
            while True:
                intento += 1
                try:
                    self._ensure_mix_iter(label)
                    it = self._mix_iters.get(label)
                    if it is None:
                        print(f"  {label}: stream muerto, recreando (intento {intento})...")
                        self._mix_iters.pop(label, None)
                        time.sleep(5)
                        continue
                    texts, appended = self._read_from_mix_iter(label, mix_bytes)
                    if not texts:
                        print(f"  {label}: vacio, esperando 5s (intento {intento})...")
                        time.sleep(5)
                        continue
                    break
                except Exception as e:
                    print(f"  {label} fallo ({e}), reintentando en 5s (intento {intento})...")
                    self._mix_iters.pop(label, None)
                    time.sleep(5)
            if label == "tuit":
                texts = [limpiar_tuit(t) for t in texts]
                texts = [t for t in texts if t]
                appended = sum(len(t.encode("utf-8")) for t in texts)
                print(f"  tuit limpio (@/http/#/espacios): {len(texts)} tuits, {appended} bytes")
            self.last_bytes[label] = appended
            out_path = mix_path or self._path
            with open(out_path, "a", encoding="utf-8") as f:
                for t in texts:
                    f.write(t)
                    f.write("\n\n")
            print(f"  Appended {appended} bytes from {label} for block {self.block_idx}")


def armar_bloque(n, ds):
    """Descarga el bloque n con dataset.py y guarda data.{n}.txt (solo corpus)."""

    ds.block_idx = n - 1  # dataset.py es base-0: data.1.txt = ventana [0,10MB), sin seek inicial
    MIN_CORPUS = int(MIN_CORPUS_MB * 1024 * 1024)
    ds._path = os.path.join(_DIR, f"wiki_block_{n}.txt")
    if os.path.exists(ds._path):
        os.remove(ds._path)
    # Un solo intento: los iteradores son persistentes (no se resetean, no se
    # re-camina desde 0). La espera infinita por mix ya garantiza el cacho.
    try:
        ds.download_block()
    except Exception as e:
        print(f"  bloque {n} fallo: {e}")
    with open(ds._path, "r", encoding="utf-8") as f:
        cuerpo = f.read()
    if len(cuerpo.encode("utf-8")) < MIN_CORPUS:
        print(f"  bloque {n} defectuoso tras 2 intentos: NO se sube, fin del loop.")
        try:
            os.remove(ds._path)
        except OSError:
            pass
        return None
    try:
        os.remove(ds._path)
    except OSError:
        pass

    out = os.path.join(_DIR, f"data.{n}.txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(cuerpo)
    total = os.path.getsize(out)
    print(f"  data.{n}.txt: {total} bytes ({total/2**20:.2f} MB)")
    print("  porcentajes -> " + " | ".join(
        f"{k} {v/total*100:.1f}%" for k, v in ds.last_bytes.items()))
    return out


def main():
    hf = HFDataManager(repo=REPO)
    if BAJAR:
        hf.download_block(BAJAR, os.path.join(_DIR, f"data.{BAJAR}.txt"))
        return
    hf.login_global()
    bloques = hf.listar_bloques() if SUBIR else []
    if bloques:
        print(f"En {hf.repo_id} hay {len(bloques)} bloques (1..{bloques[-1]}).")
        ans = input("¿Borrar TODOS y empezar de cero? (si/no) [no]: ").strip().lower()
        if ans in ("si", "sí", "s", "yes", "y"):
            for n in bloques:
                hf.borrar_bloque(n)
            print("Repo limpio.")
        else:
            print("Se conservan; se saltean los existentes.")
    if SUBIR:
        hf.ensure_repo()
        card = os.path.join(_DIR, "README.md")
        if os.path.exists(card) and not hf.readme_exists():
            hf.upload_readme(card)
        else:
            print("  README.md ya existe en el repo, no se toca.")

    ds = BlockDataset(block_idx=BLOQUE_INICIAL)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = None
        if COUNT > 0:
            ultimo = min(BLOQUE_INICIAL + COUNT, MAX_BLOQUE + 1)
        else:
            ultimo = MAX_BLOQUE + 1
        print(f"Bloques {BLOQUE_INICIAL}..{ultimo - 1} (skip existentes)")
        for n in range(BLOQUE_INICIAL, ultimo):
            if SUBIR and hf.block_exists(n):
                print(f"Bloque {n} ya existe en {hf.repo_id}, skip.")
                continue
            print(f"Bloque {n}...")
            path = armar_bloque(n, ds)
            if path is None:
                print("  streams agotados, fin.")
                break
            if fut is not None:
                fut.result()  # espera subida anterior antes de lanzar la proxima
            if SUBIR:
                # sube N en paralelo mientras el loop ya descarga N+1
                fut = pool.submit(hf.upload_block, path, n)
        if fut is not None:
            fut.result()
    if COUNT > 0 and BLOQUE_INICIAL + COUNT > MAX_BLOQUE + 1:
        print(f"Tope MAX_BLOQUE={MAX_BLOQUE} alcanzado, detenido (sin wrap).")
    print("Listo.")


if __name__ == "__main__":
    main()
