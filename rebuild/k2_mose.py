"""Asegura el K2-0.9B + MoSE-ancho en <repo>@<branch>.

Si el modelo NO esta alli: descarga IFM/K2-Horizon-0.9B original,
le pone router de ancho por capa (mose.py) y lo sube al branch.
Si ya esta: lo descarga y lo devuelve listo para train/infer.

Uso:
    model, tok = ensure_k2_mose(repo="ScortexIA/laurelia", branch="k2", token=tok)
"""

import os

import torch
from huggingface_hub import HfApi, create_repo, upload_folder, snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

from mose import wrap_k2_with_mose

BASE = "IFM/K2-Horizon-0.9B"
NEED = ("config.json", "model.safetensors")
WIDTHS = (0.25, 0.50, 0.75, 1.0)


def branch_has_model(repo: str, branch: str, token=None) -> bool:
    try:
        files = HfApi().list_repo_files(repo_id=repo, revision=branch, token=token)
    except Exception:
        return False
    return all(f in files for f in NEED)


def _load_wrapped(local_dir: str):
    from transformers import AutoConfig
    from safetensors.torch import load_file
    import glob
    tok = AutoTokenizer.from_pretrained(local_dir, trust_remote_code=True)
    # Instanciar por config (pesos random) y wrap ANTES de cargar tensores:
    # asi router.* tiene donde caer con strict=True.
    cfg = AutoConfig.from_pretrained(local_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_config(
        cfg, dtype=torch.bfloat16, trust_remote_code=True)
    wrap_k2_with_mose(model, widths=WIDTHS)
    for sf in sorted(glob.glob(os.path.join(local_dir, "*.safetensors"))):
        model.load_state_dict(load_file(sf, device="cpu"), strict=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def ensure_k2_mose(repo="ScortexIA/laurelia", branch="k2", token=None,
                   build_dir="/tmp/k2_mose_build"):
    """Devuelve (model, tok) con MoSE-ancho. Construye+sube si falta."""
    if branch_has_model(repo, branch, token):
        print(f"OK: {repo}@{branch} ya tiene el K2-MoSE, descargando...")
        local = snapshot_download(
            repo_id=repo, revision=branch, token=token,
            allow_patterns=["*.json", "*.safetensors", "*.py", "*.model",
                            "tokenizer*", "*.txt"])
        return _load_wrapped(local)

    print(f"{repo}@{branch} sin modelo: construyo desde {BASE}...")
    tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, device_map="auto", dtype=torch.bfloat16,
        low_cpu_mem_usage=True, trust_remote_code=True)
    wrap_k2_with_mose(model, widths=WIDTHS)

    os.makedirs(build_dir, exist_ok=True)
    model.save_pretrained(build_dir)
    tok.save_pretrained(build_dir)
    try:
        base_py = snapshot_download(repo_id=BASE, allow_patterns=["*.py"],
                                    local_dir=os.path.join(build_dir, "_base_py"))
        import shutil
        for f in os.listdir(os.path.join(build_dir, "_base_py")):
            if f.endswith(".py"):
                shutil.copy(os.path.join(build_dir, "_base_py", f), build_dir)
        print("  .py base incluidos")
    except Exception as e:
        print(f"  no se pudieron bajar .py base: {e}")

    api = HfApi()
    create_repo(repo_id=repo, exist_ok=True, private=False, token=token)
    try:
        api.create_branch(repo_id=repo, branch=branch, token=token)
        print(f"  rama '{branch}' creada")
    except Exception:
        pass
    upload_folder(repo_id=repo, folder_path=build_dir, revision=branch, token=token)
    print(f"Subido K2-MoSE a {repo}@{branch}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok
