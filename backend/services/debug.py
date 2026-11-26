"""요청 내용을 콘솔에 보기 좋게 찍는 개발용 도구. 이미지(base64)는 통째로 찍지 않고 요약만 찍는다.

DEBUG_SAVE=1 로 서버를 띄우면 받은 PNG를 ./debug/ 폴더에 파일로도 저장한다 (마스크를 눈으로 확인용).
"""
import base64
import os
import struct
import time
from pathlib import Path

from api.schemas import InPaintingRequest, SamplingRequest

DEBUG_SAVE = os.getenv("DEBUG_SAVE") == "1"
DEBUG_DIR = Path("debug")


def _raw_b64(b64: str) -> str:
    return b64.split(",", 1)[1] if b64.startswith("data:") else b64


def png_info(b64: str | None) -> str:
    """base64 PNG의 크기만 헤더에서 읽어 요약 (이미지 전체를 디코드하지 않음)."""
    if b64 is None:
        return "None"
    raw = _raw_b64(b64)
    size_kb = len(raw) * 3 // 4 / 1024
    head = base64.b64decode(raw[:44] + "=" * (-len(raw[:44]) % 4))
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        return f"PNG 아님? {size_kb:.1f} KB, 앞 20자={raw[:20]!r}"
    w, h = struct.unpack(">II", head[16:24])
    return f"PNG {w}x{h}, {size_kb:.1f} KB"


def curve_info(name: str, arr: list[float], steps: int) -> str:
    if not arr:
        return f"{name:<12}: (비어 있음)"
    ok = "OK" if len(arr) == steps else f"!! steps({steps})와 길이 다름"
    head = ", ".join(f"{v:.3f}" for v in arr[:3])
    tail = ", ".join(f"{v:.3f}" for v in arr[-3:])
    return f"{name:<12}: len={len(arr)} ({ok})  [{head} ... {tail}]  min={min(arr):.3f} max={max(arr):.3f}"


def print_sampling(req: SamplingRequest) -> None:
    print(f"prompt      : {req.prompt!r}")
    print(f"seed        : {req.seed}")
    print(f"steps       : {req.steps}")
    print(f"size        : {req.width} x {req.height}")
    print(curve_info("sigma_curve", req.sigma_curve, req.steps))
    print(curve_info("cfg_curve", req.cfg_curve, req.steps))
    print(curve_info("eta_curve", req.eta_curve, req.steps))


def print_inpainting(ip: InPaintingRequest) -> None:
    print(f"image_b64   : {png_info(ip.image_b64)}")
    print(f"mask_b64    : {png_info(ip.mask_b64)}")
    print(f"crop_b64    : {png_info(ip.crop_b64)}")
    print(f"crop_box    : {ip.crop_box}")


def save_debug_images(**images_b64: str | None) -> None:
    """DEBUG_SAVE=1 일 때만 ./debug/<시각>_<이름>.png 로 저장."""
    if not DEBUG_SAVE:
        return
    DEBUG_DIR.mkdir(exist_ok=True)
    stamp = time.strftime("%H%M%S")
    for name, b64 in images_b64.items():
        if b64:
            (DEBUG_DIR / f"{stamp}_{name}.png").write_bytes(base64.b64decode(_raw_b64(b64)))
            print(f"saved       : {DEBUG_DIR / f'{stamp}_{name}.png'}")