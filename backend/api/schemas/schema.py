from dataclasses import dataclass


@dataclass
class SamplingRequest:
    prompts: list[str]
    sigma_curve: list[float]
    cfg_curve: list[float]
    eta_curve: list[float]
    calculate_type:str
    seed: int | None = -1
    steps: int = 28
    width: int = 1024
    height: int = 1024
    prompt_static:bool = False
    frame_save: bool = False
    nsfw:bool = False

@dataclass
class InPaintingRequest:
    start_t : float = 0.0
    image_b64: str | None = None
    mask_b64: str | None = None
    crop_b64: str | None = None
    crop_box: list[int] | None = None


@dataclass
class SamplingResponse:                # 아직 사용 안 함 (지금은 요청만 받음)
    image_b64: str
    seed: int