from src.core import error_report

# ============================================================================= #
# mask

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def mask_generation(mask_arr, latent, thr=0.5):
    if isinstance(mask_arr, np.ndarray):
        m = torch.from_numpy(mask_arr).float()
    else:
        m = mask_arr.float()

    if m.max() > 1.0:
        m = m / 255.0

    if m.ndim == 2:
        m = m[None, None]
    elif m.ndim == 3:
        m = m[None]

    m = m.to(latent.device, latent.dtype)

    m_lat = F.adaptive_avg_pool2d(m, output_size=latent.shape[-2:])
    return (m_lat > thr).float()

# ============================================================================= #
# img resize

import cv2
import numpy as np

def arr_resize(arr, allowable_size, patch_size, compression_ratio, upscale=False) -> tuple[np.ndarray, int, int]:
    unit = patch_size * compression_ratio
    H, W = arr.shape[:2]
    long_side = max(H, W)

    def round_to_unit(x):
        return max(int(round(x / unit) * unit), unit)

    if long_side > allowable_size or upscale:
        scale = allowable_size / long_side
        if H >= W:
            new_H, new_W = round_to_unit(allowable_size), round_to_unit(W * scale)
        else:
            new_H, new_W = round_to_unit(H * scale), round_to_unit(allowable_size)
    else:
        new_H, new_W = round_to_unit(H), round_to_unit(W)

    if (new_W, new_H) != (W, H):
        interp = cv2.INTER_AREA if new_W * new_H < W * H else cv2.INTER_CUBIC
        arr = cv2.resize(arr, (new_W, new_H), interpolation=interp)

    return np.ascontiguousarray(arr), new_H, new_W

# ============================================================================= #
# img upscale

def upscale_to(img, target=1024):
    H, W = img.shape[:2]
    scale = target / max(H, W)
    if scale <= 1:
        return img
    return cv2.resize(img, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_CUBIC)

# ============================================================================= #
# img -> bytes

def img_to_bytes(img, ext=".png"):
    ok, buf = cv2.imencode(ext, img)
    if not ok:
        raise ValueError(f"encode failed: ext={ext}")
    return buf.tobytes()

# ============================================================================= #
# img merge delicated detail

def merge_img(origin_img, crop_img, box):
    x, y, w, h = box
    if crop_img.shape[:2] != (h, w):
        crop_img = cv2.resize(crop_img, (w, h), interpolation=cv2.INTER_AREA)  # 줄이기

    out = origin_img.copy()
    out[y:y+h, x:x+w] = crop_img
    return out


# ============================================================================= #
# save

import os
import json
import imageio.v2 as imageio

def save_video(path, frames, fps=16):
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    try:
        with imageio.get_writer(tmp, fps=fps, codec="libx264", quality=8) as w:
            for f in frames:
                w.append_data(np.asarray(f.convert("RGB")))
        os.replace(tmp, path)

        print(f"video : {path}")
    except Exception as e:
        error_report(e, "save_video")
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_img(path, img, format="PNG"):
    root, ext = os.path.splitext(path)
    tmp = f"{root}.tmp{ext}"
    try:
        img.save(tmp, format=format)
        os.replace(tmp, path)

        print(f"img : {path}")
    except Exception as e:
        error_report(e, "save_img")
        if os.path.exists(tmp):
            os.remove(tmp)
        raise

def save_json(path, data):
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def save_attention_plot(layer_stats_steps, save_path, h, w, max_side):
    if not layer_stats_steps:
        return

    img_tokens = max_side * max_side
    indices = (
        torch.arange(h)[:, None] * max_side
        + torch.arange(w)[None, :]
    ).flatten()

    values = {"Text → Image": [], "Image → Text": [], "Image → Image": []}

    for step_stats in layer_stats_steps:
        step_values = {key: [] for key in values}

        for stats in step_stats:
            attn = stats.get("attn_weights")

            if attn is None:
                continue

            attn = attn.float()
            idx = indices.to(attn.device)

            # Image Query → Image Key
            img_query = attn.index_select(-2, idx)
            img_img = img_query.index_select(-1, idx)
            img_to_img = img_img.sum(dim=-1).mean().item()

            # Image Query → Text Key
            img_to_text = img_query[..., img_tokens:].sum(dim=-1).mean().item()

            # Text Query → Image Key
            text_query = attn[..., img_tokens:, :]
            text_to_img = text_query.index_select(-1, idx).sum(dim=-1).mean().item()

            step_values["Text → Image"].append(text_to_img)
            step_values["Image → Text"].append(img_to_text)
            step_values["Image → Image"].append(img_to_img)

        for key in values:
            values[key].append(step_values[key])

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    steps = range(len(layer_stats_steps))

    for ax, (name, data) in zip(axes, values.items()):
        data = torch.tensor(data, dtype=torch.float32)

        for layer in range(data.shape[1]):
            ax.plot(steps, data[:, layer], label=f"Layer {layer}")

        ax.set_title(name)
        ax.set_ylabel("Attention")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=4, fontsize=8)

    axes[-1].set_xlabel("Sampling Step")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
