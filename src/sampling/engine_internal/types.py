import torch
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from src.core import state

@dataclass
class InitInput:
    steps: int
    seed: int
    H: int
    W: int
    prompts: list[str] | None = field(
        default_factory=lambda: [state.config.init_prompt]
    )

@dataclass
class InitOutput:
    cond_emb: torch.Tensor
    uncond_emb: torch.Tensor
    cond_mask: torch.Tensor
    uncond_mask: torch.Tensor
    noise: torch.Tensor
    gens: list[torch.Generator]
    seeds: list[int]
    sigmas: np.ndarray
    cfgs: np.ndarray
    etas: np.ndarray
    max_side: int

@dataclass
class InpaintInput:
    img: Image.Image
    start_t: float
    mask: Image.Image
    crop_img: Image.Image | None = None
    crop_box: list[int] | None = None
    upscale: bool = False

@dataclass(kw_only=True)
class Txt2ImgInput(InitInput):
    calculate_type: str
    sigma_curve: np.ndarray
    cfg_curve: np.ndarray
    eta_curve: np.ndarray
    frame_save: bool
    prompt_static:bool
    nsfw:bool

    def __post_init__(self):
        self.sigma_curve = np.clip(np.asarray(self.sigma_curve, dtype=np.float32), 0.001, 0.999)
        self.cfg_curve = np.asarray(self.cfg_curve, dtype=np.float32)
        self.eta_curve = np.asarray(self.eta_curve, dtype=np.float32)
    
@dataclass
class InpaintCtx:
    z0: torch.Tensor
    mask: torch.Tensor