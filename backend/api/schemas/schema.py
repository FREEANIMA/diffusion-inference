from dataclasses import dataclass


@dataclass
class SamplingRequest:
    prompts: list[str]
    sigma_curve: list[float]
    cfg_curve: list[float]
    eta_curve: list[float]
    calculate_type:str
    seed: int
    steps: int
    width: int
    height: int
    prompt_static:bool
    frame_save: bool
    nsfw:bool

@dataclass
class InPaintingRequest:
    start_t : float
    image_b64: str | None = None
    mask_b64: str | None = None
    crop_b64: str | None = None
    crop_box: list[int] | None = None


@dataclass
class SamplingResponse:
    image_b64: str
    seed: int