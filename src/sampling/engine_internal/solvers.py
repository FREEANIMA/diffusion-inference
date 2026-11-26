import math


def euler(x, v, sigma, sigma_next):
    return x + (sigma_next - sigma) * v


def heun_step(model_fn, x, sigma, sigma_next):
    v1 = model_fn(x, sigma)
    x_euler = x + (sigma_next - sigma) * v1
    if sigma_next == 0:
        return x_euler
    v2 = model_fn(x_euler, sigma_next)
    return x + (sigma_next - sigma) * 0.5 * (v1 + v2)


def midpoint_step(model_fn, x, sigma, sigma_next):
    sigma_mid = 0.5 * (sigma + sigma_next)
    v1 = model_fn(x, sigma)
    x_mid = x + (sigma_mid - sigma) * v1
    v2 = model_fn(x_mid, sigma_mid)
    return x + (sigma_next - sigma) * v2


def sigma_lo_of(sigma_next, eta):
    """eta=0이면 sigma_next, eta=1이면 0."""
    return float(sigma_next * (1.0 - eta))


def renoise(x_lo, z, sigma_lo, sigma_next):
    """
    x_lo: sigma_lo 상태의 샘플 (반드시 (1-sigma_lo)*x0 + sigma_lo*eps 형태)
    z: 새 노이즈 (N(0,I))
    """
    sigma_lo = float(sigma_lo)
    sigma_next = float(sigma_next)
    a = (1.0 - sigma_next) / (1.0 - sigma_lo)
    var = sigma_next ** 2 - (a * sigma_lo) ** 2
    noise_scale = max(var, 0.0) ** 0.5
    return a * x_lo + noise_scale * z


def dpmpp_2m(x, x0, x0_prev, sigma, sigma_next, h_prev):
    """DPM++ 2M (x0 기반)."""
    alpha = 1.0 - sigma
    alpha_n = 1.0 - sigma_next
    ratio = sigma_next / sigma

    D = x0
    h = None
    if 0.0 < sigma < 1.0 and sigma_next > 0.0:
        h = math.log(alpha_n / sigma_next) - math.log(alpha / sigma)
        if x0_prev is not None and h_prev is not None and h_prev > 0 and h != 0:
            r = h_prev / h
            D = (1.0 + 1.0 / (2.0 * r)) * x0 - (1.0 / (2.0 * r)) * x0_prev

    x_next = ratio * x + (alpha_n - alpha * ratio) * D
    return x_next, h