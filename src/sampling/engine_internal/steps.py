import torch
from tqdm.auto import tqdm

from src.network import MMDIT
from src.core import state
from .types import Txt2ImgInput, InitOutput, InpaintCtx
from .solvers import euler, renoise, sigma_lo_of, heun_step, dpmpp_2m


@torch.inference_mode()
@torch.amp.autocast(device_type=state.config.device, dtype=state.config.torch_dtype)
def one_steps(model: MMDIT, x, init_out: InitOutput, sigma, cfg) -> torch.Tensor:
    x_in = x.to(state.config.torch_dtype)
    B, _, _, _ = x_in.shape

    sigma_1d = torch.full((2 * B,), sigma, device=state.config.device, dtype=torch.float32)

    out, _ = model.forward(
        x=torch.cat([x_in, x_in], dim=0),
        sigma=sigma_1d,
        txt_emb=torch.cat([init_out.cond_emb, init_out.uncond_emb], dim=0),
        txt_mask=torch.cat([init_out.cond_mask, init_out.uncond_mask], dim=0),
        max_side=init_out.max_side,
    )

    c, u = out.float().chunk(2, dim=0)
    v = u + cfg * (c - u)
    return v


def sampling(model, txt2img_input: Txt2ImgInput, init_out: InitOutput,
             inpaint_ctx: InpaintCtx = None, on_step=None):
    model.eval()
    sigmas = init_out.sigmas
    n = len(sigmas) - 1
    calc = txt2img_input.calculate_type

    latents = []
    sigmas_steps = []
    noise = init_out.noise
    gens = init_out.gens

    if inpaint_ctx is not None:
        z0 = inpaint_ctx.z0.to(device=noise.device, dtype=noise.dtype)
        mask = inpaint_ctx.mask.to(device=noise.device, dtype=noise.dtype)

        def clamp_known(x, s):
            return mask * x + (1 - mask) * ((1 - s) * z0 + s * noise)
    else:
        clamp_known = None

    x = noise if clamp_known is None else clamp_known(noise, float(sigmas[0]))

    # dpmpp_2m 상태
    x0_prev, h_prev = None, None

    pbar = tqdm(range(n), desc=f"sampling[{calc}]", total=n,
                dynamic_ncols=True, leave=True)

    for i in pbar:
        sigma = float(sigmas[i])
        sigma_next = float(sigmas[i + 1])
        cfg = float(init_out.cfgs[i])
        eta = float(init_out.etas[i])
        last = (i == n - 1)

        # eta 정규화: 아주 작으면 0으로
        if eta < 1e-6:
            eta = 0.0

        first_v = []

        def model_fn(x_in, s, cfg=cfg):
            if clamp_known is not None:
                x_in = clamp_known(x_in, s)
            v = one_steps(model, x_in, init_out, s, cfg)
            if not first_v:
                first_v.append((x_in, v))
            return v

        v = model_fn(x, sigma)

        # eta/noise 준비
        use_noise = (eta > 1e-6) and not last
        if use_noise:
            sigma_lo = sigma_lo_of(sigma_next, eta)   # float
            z = torch.cat([
                torch.randn((1, *x.shape[1:]), generator=g_, device=x.device, dtype=x.dtype)
                for g_ in gens
            ], dim=0)
        else:
            sigma_lo = sigma_next
            z = None

        # 진행바 postfix
        pbar.set_postfix({
            "s": f"{sigma:.3f}->{sigma_next:.3f}",
            "cfg": f"{cfg:.1f}",
            "eta": f"{eta:.3f}",
            "x_std": f"{x.std().item():.2f}",
            "v_std": f"{v.std().item():.2f}",
        }, refresh=False)

        # --- 스텝 ---
        if calc == "euler":
            if use_noise:
                # x_lo를 이론적 형태로 재구성 (renoise 정확도 ↑)
                x0 = x - sigma * v
                eps = x + (1.0 - sigma) * v
                x_lo = (1.0 - sigma_lo) * x0 + sigma_lo * eps
                x = renoise(x_lo, z, sigma_lo, sigma_next)
            else:
                x = euler(x, v, sigma, sigma_next)

        elif calc == "heun":
            if last:
                x = euler(x, v, sigma, sigma_next)
            else:
                if use_noise:
                    # heun 1단계: sigma -> sigma_lo
                    x0 = x - sigma * v
                    eps = x + (1.0 - sigma) * v
                    x_e = (1.0 - sigma_lo) * x0 + sigma_lo * eps
                    v2 = model_fn(x_e, sigma_lo)
                    # heun 평균으로 x_lo 재구성
                    v_avg = 0.5 * (v + v2)
                    x0_lo = x_e - sigma_lo * v_avg
                    eps_lo = x_e + (1.0 - sigma_lo) * v_avg
                    x_lo = (1.0 - sigma_lo) * x0_lo + sigma_lo * eps_lo  # = x_e 그대로
                    x = renoise(x_e, z, sigma_lo, sigma_next)
                    del v2, x0_lo, eps_lo
                else:
                    x = heun_step(model_fn, x, sigma, sigma_next)

        elif calc == "dpmpp_2m":
            x0_cur = x - sigma * v
            x, h = dpmpp_2m(x, x0_cur, x0_prev, sigma, sigma_next, h_prev)
            if use_noise:
                # dpmpp 결과에 renoise 적용 (x_lo 재구성)
                x0 = x - sigma_next * v  # 근사
                eps = x + (1.0 - sigma_next) * v
                x_lo = (1.0 - sigma_lo) * x0 + sigma_lo * eps
                x = renoise(x_lo, z, sigma_lo, sigma_next)
                x0_prev, h_prev = None, None
            else:
                x0_prev, h_prev = x0_cur, h

        else:
            raise ValueError(f"Unsupported sampling method: {calc}")

        del z

        # x0_hat 기록
        x_in0, v0 = first_v[0]
        x0_hat = x_in0 - sigma * v0

        if clamp_known is not None:
            x = clamp_known(x, sigma_next)

        sigmas_steps.append(sigma_next)
        latents.append(x0_hat)

        # 스텝 완료 로그
        tqdm.write(
            f"[{i:02d}] s={sigma:.3f}->{sigma_next:.3f} cfg={cfg:.1f} eta={eta:.3f} "
            f"noise={'Y' if use_noise else 'N'} | "
            f"x_in.std={x_in0.std().item():.3f} x_in.max={x_in0.abs().max().item():.2f} | "
            f"v.std={v0.std().item():.3f} v.max={v0.abs().max().item():.2f} | "
            f"x0hat.std={x0_hat.std().item():.3f} | "
            f"x_next.std={x.std().item():.3f} x_next.max={x.abs().max().item():.2f} | "
            f"nan={bool(torch.isnan(x).any())}"
        )

        if on_step:
            on_step(i + 1, n)

    pbar.close()
    return x, latents, sigmas_steps