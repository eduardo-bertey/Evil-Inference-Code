"""Helpers de HuggingFace (mínimo, sin clases)."""

import getpass
import os


def hf_token(repo):
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if tok:
        print("HF token desde entorno")
        return tok
    print(f"\nNo HF token found. Enter token for {repo} (write access):")
    tok = getpass.getpass("Token: ").strip()
    try:
        from huggingface_hub import login
        login(token=tok, add_to_git_credential=False)
        print("HF token aplicado globalmente para descargas")
    except Exception as e:
        print(f"  login fallo ({e}), sigo con token en memoria")
    return tok


def hf_api(token):
    from huggingface_hub import HfApi
    return HfApi(token=token)


def hf_ensure_repo_branch(api, repo, branch, token):
    from huggingface_hub import create_repo
    try:
        create_repo(repo_id=repo, exist_ok=True, private=False, token=token)
    except Exception:
        pass
    if branch not in ("main", "master"):
        try:
            api.create_branch(repo_id=repo, branch=branch, token=token)
            print(f"Rama '{branch}' creada en {repo}")
        except Exception:
            pass


def hf_download(repo, branch, filename, local_path, token):
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo, filename=filename,
                               revision=branch, token=token)
        import shutil
        shutil.copy2(path, local_path)
        print(f"Descargado {filename} de {repo}@{branch}")
        return True
    except Exception as e:
        print(f"No se pudo descargar {filename}: {e}")
        return False


def hf_upload_file(api, repo, branch, path, path_in_repo, token, msg):
    api.upload_file(path_or_fileobj=path, path_in_repo=path_in_repo,
                    repo_id=repo, revision=branch, token=token,
                    commit_message=msg)
    print(f"  subido {path_in_repo} -> {repo}@{branch}")


def hf_upload_folder(repo, branch, folder, token, msg):
    from huggingface_hub import upload_folder
    upload_folder(repo_id=repo, folder_path=folder, revision=branch,
                  token=token, commit_message=msg)
    print(f"Subido {folder} -> {repo}@{branch}")
