
import torch
from tqdm.auto import tqdm

from src.network import MMDIT
from src.core import state
from .types import Txt2ImgInput, InitOutput, InpaintCtx
from .solvers import (
    euler, midpoint_step, heun_step, rk4_step,
    ab2_step, renoise, sigma_lo_of, dpmpp_2m,
)

@torch.inference_mode()
@torch.amp.autocast(device_type=state.config.device, dtype=state.config.torch_dtype)
def one_steps(model: MMDIT, x, init_out: InitOutput, sigma, cfg) -> tuple[torch.Tensor, dict]:
    B = x.shape[0]
    x = x.to(state.config.torch_dtype)

    out, layers_state = model.forward(
        x=torch.cat([x, x]),
        sigma=torch.full((2 * B,), sigma, device=x.device, dtype=torch.float32),
        txt_emb=torch.cat([init_out.cond_emb, init_out.uncond_emb]),
        txt_mask=torch.cat([init_out.cond_mask, init_out.uncond_mask]),
        max_side=init_out.max_side,
        collects_stats=state.config.attn_report,
        attn_collects_stats=state.config.attn_report
    )

    c, u = out.float().chunk(2)
    return u + cfg * (c - u), layers_state


def sampling(model: MMDIT, txt2img_input: Txt2ImgInput, init_out: InitOutput, inpaint_ctx: InpaintCtx = None, on_step=None, on_preview=None, preview_every=4):
    model.eval()
    sigmas = init_out.sigmas
    n = len(sigmas) - 1
    calc = txt2img_input.calculate_type

    solvers = {
        "euler", "midpoint", "heun", "rk4", "ab2", "dpmpp_2m"
    }
    if calc not in solvers:
        raise ValueError(f"Unsupported sampling method: {calc}")

    latents, sigmas_steps, layer_stats_steps = [], [], []
    noise, gens = init_out.noise, init_out.gens

    # Inpainting
    if inpaint_ctx is not None:
        z0 = inpaint_ctx.z0.to(device=noise.device, dtype=noise.dtype)
        mask = inpaint_ctx.mask.to(device=noise.device, dtype=noise.dtype)

        def known_at(s):
            return (1.0 - s) * z0 + s * noise

        def clamp_known(x, s):
            return mask * x + (1.0 - mask) * known_at(s)

        # 마스크 영역도 start sigma 수준의 (z0 + noise)로 시작
        x = known_at(float(sigmas[0]))
    else:
        clamp_known = None
        x = noise

    # Solver states
    x0_prev, dpmpp_h_prev = None, None
    v_prev, ab2_h_prev = None, None

    pbar = tqdm(range(n), desc=f"sampling[{calc}]", total=n, dynamic_ncols=True, leave=True)

    for i in pbar:
        sigma, sigma_next = float(sigmas[i]), float(sigmas[i + 1])
        cfg, eta = float(init_out.cfgs[i]), float(init_out.etas[i])
        last = i == n - 1

        if not 0.0 <= eta <= 1.0:
            raise ValueError(f"eta must be between 0 and 1: {eta}")
        if eta < 1e-6:
            eta = 0.0

        # Model function
        first_v, first_stats = [], []

        def model_fn(x_in, s):
            if clamp_known is not None:
                x_in = clamp_known(x_in, s)

            v, stats = one_steps(model, x_in, init_out, s, cfg)

            if not first_v:
                first_v.append((x_in, v))
                first_stats.append(stats)

            return v

        v = model_fn(x, sigma)

        # Stochastic transition
        use_noise = eta > 1e-6 and not last
        sigma_lo = sigma_lo_of(sigma_next, eta) if use_noise else sigma_next

        z = None
        if use_noise:
            z = torch.cat([
                torch.randn((1, *x.shape[1:]), generator=g, device=x.device, dtype=x.dtype)
                for g in gens
            ])

        pbar.set_postfix({
            "s": f"{sigma:.3f}->{sigma_next:.3f}",
            "cfg": f"{cfg:.1f}",
            "eta": f"{eta:.3f}",
            "x_std": f"{x.std().item():.2f}",
            "v_std": f"{v.std().item():.2f}",
        }, refresh=False)

        # Solver
        sigma_target = sigma_lo if use_noise else sigma_next

        if calc == "euler":
            x_target = euler(x, v, sigma, sigma_target)
        elif calc == "midpoint":
            x_target = midpoint_step(model_fn, x, sigma, sigma_target, v)
        elif calc == "heun":
            x_target = heun_step(model_fn, x, sigma, sigma_target, v)
        elif calc == "rk4":
            x_target = rk4_step(model_fn, x, sigma, sigma_target, v)
        elif calc == "ab2":
            x_target = ab2_step(x, v, v_prev, sigma, sigma_target, ab2_h_prev)
        elif calc == "dpmpp_2m":
            x0_cur = x - sigma * v
            x_target, h = dpmpp_2m(x, x0_cur, x0_prev, sigma, sigma_target, dpmpp_h_prev)

            x0_prev, dpmpp_h_prev = (None, None) if use_noise else (x0_cur, h)

        # Noise transition
        if use_noise:
            x = renoise(x_target, z, sigma_lo, sigma_next)
            v_prev, ab2_h_prev = None, None

            if calc != "dpmpp_2m":
                x0_prev, dpmpp_h_prev = None, None
        else:
            x = x_target

            if calc == "ab2":
                v_prev, ab2_h_prev = v, sigma_target - sigma
            elif calc != "dpmpp_2m":
                v_prev, ab2_h_prev = None, None

        del z

        # x0 prediction
        x_in0, v0 = first_v[0]
        x0_hat = x_in0 - sigma * v0

        # Inpainting clamp
        if clamp_known is not None:
            x = clamp_known(x, sigma_next)

        # Preview
        if on_preview and (i % preview_every == 0 or last):
            on_preview(i + 1, n, x0_hat[:1].detach())

        # History
        sigmas_steps.append(sigma_next)
        latents.append(x0_hat)

        step_stats = {}

        for layer, stats in first_stats[0].items():
            step_stats[layer] = {
                "attn": {
                    key: value.detach().cpu()
                    for key, value in stats["attn"].items()
                } if stats.get("attn") is not None else None
            }

        layer_stats_steps.append(step_stats)

        # Logging
        # tqdm.write(
        #     f"[{i:02d}] s={sigma:.3f}->{sigma_next:.3f} cfg={cfg:.1f} eta={eta:.3f} "
        #     f"noise={'Y' if use_noise else 'N'} | "
        #     f"x_in.std={x_in0.std().item():.3f} x_in.max={x_in0.abs().max().item():.2f} | "
        #     f"v.std={v0.std().item():.3f} v.max={v0.abs().max().item():.2f} | "
        #     f"x0hat.std={x0_hat.std().item():.3f} | "
        #     f"x_next.std={x.std().item():.3f} x_next.max={x.abs().max().item():.2f} | "
        #     f"nan={bool(torch.isnan(x).any())}"
        # )

        if on_step:
            on_step(i + 1, n)

    pbar.close()
    return x, latents, sigmas_steps, layer_stats_steps
