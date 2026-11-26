import os
import time
import traceback
from src.core import shared

def log(console, msg):
    if console is None:
        print(msg)
    else:
        console.put(msg)

def error_report(error=None, name=None):
    tag = f"{name}_" if name else ""
    path = os.path.join(shared.BASE_ROOT, "error_report", f"error_{tag}{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1_000_000}.txt")

    body = traceback.format_exc()
    if body.strip() == "NoneType: None":
        body = ""
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{type(error).__name__ if error else ''}: {error if error else ''}\n\n{body}")
    return path

import dataclasses
import numpy as np
import torch
from PIL import Image

def summ(v) -> str:
    if isinstance(v, torch.Tensor):
        return f"Tensor{tuple(v.shape)} {str(v.dtype).replace('torch.', '')} {v.device}"
    if isinstance(v, np.ndarray):
        return f"ndarray{v.shape} {v.dtype}"
    if isinstance(v, Image.Image):
        return f"PIL {v.size[0]}x{v.size[1]} {v.mode}"  # (W x H)
    if isinstance(v, torch.Generator):
        return f"Generator({v.device})"
    if isinstance(v, str):
        return repr(v if len(v) <= 40 else v[:37] + "...")
    if isinstance(v, (list, tuple)):
        if len(v) <= 4 and all(isinstance(x, (int, float, str)) for x in v):
            return f"{type(v).__name__}{[summ(x) if isinstance(x, str) else x for x in v]}"
        inner = summ(v[0]) if v else "-"
        return f"{type(v).__name__}[len={len(v)}] of {inner}"
    return repr(v)

def dbg(obj, title: str | None = None):
    title = title or type(obj).__name__
    print(f"── {title} " + "─" * max(0, 50 - len(title)))
    if dataclasses.is_dataclass(obj):
        w = max(len(f.name) for f in dataclasses.fields(obj))
        for f in dataclasses.fields(obj):
            print(f"  {f.name:<{w}} : {summ(getattr(obj, f.name))}")
    else:
        print(f"  {summ(obj)}")