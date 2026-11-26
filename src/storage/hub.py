import os
import hashlib

from huggingface_hub import hf_hub_download, HfApi
from huggingface_hub.utils import EntryNotFoundError, RepositoryNotFoundError
from src.core import Config, error_report

def get_file_hash(path: str) -> str:
    sha256 = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()

def download_model(config: Config, filename) -> str | None:
    try:
        repo_id = f"{config.CKPT_REPO_ID}/VER{config.ver_number}"
        local_path = os.path.join(config.CKPT_LOCAL_DIR, filename)

        if os.path.isfile(local_path):
            api = HfApi()
            info = api.get_paths_info(
                repo_id=repo_id,
                paths=[filename],
                repo_type="model"
            )

            if info:
                remote_file = info[0]

                if remote_file.lfs:
                    remote_hash = remote_file.lfs["sha256"]
                    local_hash = get_file_hash(local_path)
                else:
                    sha1 = hashlib.sha1()
                    size = os.path.getsize(local_path)
                    sha1.update(f"blob {size}\0".encode())

                    with open(local_path, "rb") as f:
                        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
                            sha1.update(chunk)

                    local_hash = sha1.hexdigest()
                    remote_hash = remote_file.blob_id

                if local_hash != remote_hash:
                    print(f"\n[UPDATE] {filename}")
                    print("Local file and remote file hashes differ.")

                    if input("Download the updated file? (y/N): ").strip().lower() != "y":
                        return local_path

                else:
                    return local_path

        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="model",
            local_dir=config.CKPT_LOCAL_DIR
        )

        return path

    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        error_report(e, "download_failed")
        return None

def download_meta(config:Config, filename) -> str | None:
    try:
        path = hf_hub_download(
            repo_id=config.META_REPO_ID,
            filename=filename,
            repo_type="dataset",
            local_dir=config.DATA_LOCAL_DIR
        )
        return path
    except (EntryNotFoundError, RepositoryNotFoundError) as e:
        return None