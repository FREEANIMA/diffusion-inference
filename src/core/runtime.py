from dataclasses import fields
import json, os, glob
from .config import Config
from .shared import BASE_ROOT
from version import VERSION

def prepared_dir(config:Config):
    os.makedirs(config.GENERATION_DIR, exist_ok=True)
    os.makedirs(config.INPAINTING_DIR, exist_ok=True)
    os.makedirs(config.DETAIL_DIR, exist_ok=True)
    os.makedirs(config.ERROR_REPORT, exist_ok=True)
    os.makedirs(config.CKPT_LOCAL_DIR, exist_ok=True)

def version_to_number(version: str) -> int:
    _, minor, patch = map(int, version.split("."))
    return minor * 10 + patch

from dataclasses import fields

def print_config(config, title="Config", max_val_len=60):
    rows = []
    for f in fields(config):
        val = getattr(config, f.name)
        val_str = repr(val)
        if len(val_str) > max_val_len:
            val_str = val_str[:max_val_len - 3] + "..."
        rows.append((f.name, val_str, type(val).__name__))

    if not rows:
        print(f"({title} is empty)")
        return

    # 컬럼 폭 계산
    w_name = max(len("Field"), max(len(r[0]) for r in rows))
    w_val  = max(len("Value"), max(len(r[1]) for r in rows))
    w_type = max(len("Type"),  max(len(r[2]) for r in rows))

    # 내부 폭 (좌우 공백 1칸씩 포함)
    inner = w_name + w_val + w_type + 8

    # 타이틀 라인
    title_str = f" {title} "
    if len(title_str) >= inner:
        top = "┌" + title_str + "┐"
    else:
        left = (inner - len(title_str)) // 2
        right = inner - len(title_str) - left
        top = "┌" + "─" * left + title_str + "─" * right + "┐"

    header = f"│ {'Field':<{w_name}} │ {'Value':<{w_val}} │ {'Type':<{w_type}} │"
    sep    = f"├─{'─' * w_name}─┼─{'─' * w_val}─┼─{'─' * w_type}─┤"
    bottom = f"└─{'─' * w_name}─┴─{'─' * w_val}─┴─{'─' * w_type}─┘"

    print(top)
    print(header)
    print(sep)
    for name, val, typ in rows:
        print(f"│ {name:<{w_name}} │ {val:<{w_val}} │ {typ:<{w_type}} │")
    print(bottom)

def config_settings():
    print(f"Version : {VERSION}")
    ver_number = version_to_number(VERSION)
    pattern = os.path.join(BASE_ROOT, "configs", f"version_{ver_number}", "*.json")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"json 파일이 없습니다: {pattern}")

    config_dict = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            config_dict.update(json.load(f))

    def pick(cls, params):
        names = {f.name for f in fields(cls)}
        return {k: v for k, v in params.items() if k in names}

    config = Config(**pick(Config, config_dict))


    print_config(config, title=f"Config (v{VERSION})")

    return config