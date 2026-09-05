"""HuggingFace dataset repo: token (como train de laurelia), subir/bajar bloques.

Repo destino: data-fine-es (namespace = tu usuario, se resuelve con whoami).
Bloques: data.1.txt, data.2.txt, ...
"""

import getpass
import os

from huggingface_hub import HfApi, create_repo, hf_hub_download


class HFDataManager:
    def __init__(self, repo: str = "data-fine-es", token: str | None = None):
        self._repo_short = repo
        self._token = token
        self._api = None
        self._repo_id = None

    def _get_token(self) -> str:
        if self._token:
            return self._token
        env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        if env_token:
            self._token = env_token
            return env_token
        print("\nNo HF token found. Enter token (write access):")
        token = getpass.getpass("Token: ").strip()
        if not token:
            raise ValueError("Token required for HuggingFace operations")
        self._token = token
        return token

    def _get_api(self) -> HfApi:
        if self._api is None:
            self._api = HfApi(token=self._get_token())
        return self._api

    @property
    def repo_id(self) -> str:
        if self._repo_id is None:
            if "/" in self._repo_short:
                self._repo_id = self._repo_short
            else:
                user = self._get_api().whoami()["name"]
                self._repo_id = f"{user}/{self._repo_short}"
        return self._repo_id

    def ensure_repo(self):
        create_repo(repo_id=self.repo_id, repo_type="dataset",
                    exist_ok=True, private=False, token=self._get_token())
        print(f"Repo dataset listo: {self.repo_id} (publico)")

    def block_exists(self, n: int) -> bool:
        try:
            return self._get_api().file_exists(
                repo_id=self.repo_id, filename=f"data.{n}.txt", repo_type="dataset")
        except Exception:
            return False

    def upload_block(self, local_path: str, n: int):
        self._get_api().upload_file(
            path_or_fileobj=local_path,
            path_in_repo=f"data.{n}.txt",
            repo_id=self.repo_id,
            repo_type="dataset",
            token=self._get_token(),
            commit_message=f"bloque {n}",
        )
        print(f"  subido {self.repo_id}/data.{n}.txt")

    def download_block(self, n: int, local_path: str) -> bool:
        try:
            path = hf_hub_download(repo_id=self.repo_id, filename=f"data.{n}.txt",
                                   repo_type="dataset", token=self._get_token())
            import shutil
            shutil.copy2(path, local_path)
            print(f"  bajado data.{n}.txt -> {local_path}")
            return True
        except Exception as e:
            print(f"  no se pudo bajar data.{n}.txt: {e}")
            return False
