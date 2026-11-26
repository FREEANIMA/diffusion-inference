import numpy as np
from src.core import Config

# u = np.linspace(0.0, 1.0, txt2img_input.steps + 1)
# sigmas = sigma_cdf(u, config)[::-1].copy()

# c = np.linspace(0.0, 1.0, txt2img_input.steps + 1)
# cfgs = cfg_schedular(c, txt2img_input.cfg_scale, config)[::-1].copy()

# d = np.linspace(0.0, 1.0, txt2img_input.steps + 1)
# etas = eta_schedular(d, txt2img_input.eta, config)[::-1].copy()


def sigma_cdf(x, config:Config) -> np.ndarray:
    z = np.clip(config.beta + config.gamma * x, -1.0, 1.0)
    y = config.alpha * np.arccos(z) + config.zeta
    y = np.clip(y, config.low_sigma, config.high_sigma)

    return y

def cfg_schedular(x, cfg_scale, config:Config) -> np.ndarray:
    z = np.pi*(config.cfg_alpha + config.cfg_beta * x) / config.cfg_gamma
    y = np.sin(z)
    y = np.clip(y, 0, 1)

    return y * cfg_scale

def eta_schedular(x, eta, config:Config) -> np.ndarray:
    z = np.pi*(config.eta_alpha + config.eta_beta * x) / config.eta_gamma
    y = np.sin(z)
    y = np.clip(y, 0, 1)

    return y * eta