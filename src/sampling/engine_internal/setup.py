import random
import torch
import numpy as np

from src.external import TE
from src.core import Config
from .types import Txt2ImgInput, InitOutput
# from .trajectory import sigma_cdf, cfg_schedular, eta_schedular

@torch.inference_mode()
def setup_init(model:TE, config:Config, txt2img_input:Txt2ImgInput) -> InitOutput:
    H = txt2img_input.H
    W = txt2img_input.W 

    if max(H, W) > 1024:
        allowable_size = 2048
    else:
        allowable_size = 1024
        
    max_side = allowable_size // (config.patch_size * config.compression_ratio)

    B = len(txt2img_input.prompts)
    prompts_batch = txt2img_input.prompts
    init_prompt_batch = [config.init_prompt] * len(txt2img_input.prompts)
    text_batch = prompts_batch + init_prompt_batch

    emb, mask = model.encoder(captions=text_batch)
    cond_emb, uncond_emb = emb[:B], emb[B:]
    cond_mask, uncond_mask = mask[:B], mask[B:]

    seed = random.randint(0, 2**32 - 1) if txt2img_input.seed == -1 else txt2img_input.seed
    rng = random.Random(seed)

    seeds = [seed] + [rng.randint(0, 2**32 - 1) for _ in range(B - 1)]
    gens = [torch.Generator(device=config.device).manual_seed(s) for s in seeds]

    f = config.compression_ratio * config.patch_size

    noise = torch.cat([
        torch.randn(
            (1, config.in_channels, txt2img_input.H // f, txt2img_input.W // f),
            generator=g_,
            device=config.device,
            dtype=torch.float32,
        )
        for g_ in gens
    ], dim=0)

    return InitOutput(
        cond_emb=cond_emb,
        uncond_emb=uncond_emb,
        cond_mask=cond_mask,
        uncond_mask=uncond_mask,
        noise=noise,
        gens=gens,
        seeds=seeds,
        sigmas=txt2img_input.sigma_curve,
        cfgs=txt2img_input.cfg_curve,
        etas=txt2img_input.eta_curve,
        max_side=max_side,
    )