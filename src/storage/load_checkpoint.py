from typing import Any
import os
import json
import torch

from src.core import log, Config
from .hub import download_model
from src.network import MMDIT
from safetensors import safe_open
from safetensors.torch import load_file

def load_checkpoint_bin(config: Config, filename: str, path=None, console=None) -> dict[str, Any] | None:
    if path is None or not os.path.isfile(path):
        log(console, "Downloading Model to local from hugging face")
        path = download_model(config=config, filename=filename)

    if path is None:
        log(console, "Model not found")
        return None

    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def load_ema_only(config: Config, network: MMDIT, console=None) -> MMDIT:
    path = os.path.join(config.CKPT_LOCAL_DIR, config.EMA_FILENAME)
    if not os.path.isfile(path):
        log(console, "Downloading Model to local from hugging face")
        path = download_model(config=config, filename=config.EMA_FILENAME)

    if path is None or not os.path.isfile(path):
        msg = "Model not found on Hugging Face. Please contact the administrator."
        log(console, msg)
        raise FileNotFoundError(msg)

    log(console, f"loading safetensors: {path}")
    with safe_open(path, framework="pt") as f:
        model_cfg = json.loads(f.metadata()["config"])

    prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(config.torch_dtype)
    try:
        with torch.device(config.device):
            model: MMDIT = network(**model_cfg)
    finally:
        torch.set_default_dtype(prev_dtype)

    own = model.state_dict()
    with safe_open(path, framework="pt", device="cpu") as f:
        keys = set(f.keys())
        missing = set(own.keys()) - keys
        unexpected = keys - set(own.keys())
        if missing or unexpected:
            raise KeyError(f"state_dict mismatch. missing={list(missing)[:5]}, unexpected={list(unexpected)[:5]}")
        with torch.no_grad():
            for k in keys:
                own[k].copy_(f.get_tensor(k))

    return model.eval()


def load_llm_model(config: Config, network, vocab_size, console=None):
    path = os.path.join(config.CKPT_LOCAL_DIR, config.LLM_FILENAME)
    ckpt = load_checkpoint_bin(config=config, path=path, filename=config.LLM_FILENAME, console=console)
    if ckpt is None:
        msg = "Model not found on Hugging Face. Please contact the administrator."
        log(console, msg)
        raise FileNotFoundError(msg)

    model = network(**{**ckpt["config"], "vocab_size": vocab_size})
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()})

    return model.to(device=config.device, dtype=config.torch_dtype).eval()