## Update — 2026.11.20 Sampling: ETA injection equation selection & per-step latent → frame save

<img src="assets/eta_frame.png" alt="update" width="50%">

### ETA mode

`eta_mode = cos | ln | sin`

`ln`: `eta * ln(2 / (1 + x)) / ln 2`

<img src="assets/graph.png" alt="graph" width="30%">

Code: `src.utilities.predict`

- `Eta`: 0 is deterministic. The larger it is, the more fresh noise is injected at each step.
- `Eta mode`: schedule of the injected noise over the steps. Selectable in all three tabs.

### Frame save

Collect the latent at every step, decode all of them into images, and save them as an MP4.

<img src="assets/detailer_process.gif" controls width="600">

Toggle with **Frame save** in the Sampling panel.

### Batch

Generate 1–8 images at once. When Batch is greater than 1, numbered buttons under the result switch between images. Hover over an image to see its seed.

---

## Update — 2026.11.19 Sampling

- **Quantization**: optional quantized model loading, chosen at startup (`use quantization? y/n`). Lower VRAM usage; works with the block-offload mode.
- **Batch generation**: generate up to 8 images per run (Text2img / Inpainting). Results appear with numbered buttons to switch between images.
- **Detailer batch**: each generated patch is merged into a copy of the original image and saved separately. The original is never modified.
- **API**: `/api/generate` now accepts `batch` and returns `images` (array) and `path` (list) instead of a single `image`.

## Update — 2026.11.17 Sampling

### Sampling Improvements

- **DDIM sampler removed**
  - Removed the standalone DDIM sampler due to unsatisfactory generation quality in the current implementation.

- **Noise Injection added to Euler and Heun**
  - Introduced additional noise injection during the sampling process.
  - Prevents the sampling trajectory from becoming overly constrained by the initial latent structure.
  - Allows the model to reconstruct local features while better utilizing surrounding image context.
  - Improves structural consistency across neighboring regions.

- **`eta` parameter (noise injection strength)**
  - Each step integrates only down to `σ_lo = σ_next × (1 − eta)`, then re-noises back up to `σ_next` with fresh noise.
  - `eta = 0`: no noise injection. Identical to the original Euler / Heun / DDIM behavior.
  - Higher `eta`: a larger back-and-forth interval per step, so more of the previous structure is re-noised and re-decided.
  - Too high: details and composition can drift between steps, and Heun's second-order correction covers a shorter interval, so its benefit shrinks.
  - Suggested starting range: `0.1 ~ 0.3` (default `0.3` for DDPM_PLUS). Tune per step count and CFG scale.
  - The final step never injects noise, regardless of `eta`.

### Quality Improvements

- Improved eye and facial detail reconstruction.
- Better consistency between left and right facial features.
- Improved transitions between modified and preserved regions.
- Reduced local structural artifacts.
- Significant quality improvements in realistic image generation and inpainting.

### Available Samplers

| Sampler | Status | Uses `eta` |
|---|---|---|
| Euler + Noise Injection | Supported | Yes |
| Heun + Noise Injection | Supported | Yes |
| DDPM_PLUS | Supported | Yes |
| DDIM | Removed | - |

### Notes

Noise injection improves the model's ability to reconstruct local structures instead of relying too heavily on the initial latent representation.

This is particularly beneficial for realistic image generation, where fine details and spatial consistency are important.

## Inference Requirements
- **Inference:** Min VRAM 15GB
- **Storage:** Min 30GB free disk space (excluding `requirements.txt` packages)
- **Environment:** Local & Google Colab supported

## Update 2026 11 16 Sampling GPU Optimize
### CPU Offloading Mode

If your GPU has less than 15GB VRAM, run inference in CPU offloading mode.
The model stays on CPU; transformer blocks are streamed to the GPU in chunks.

On startup, choose `n`:

```
model upload on gpu (y or n): n
```

Then set the chunk size (layers loaded onto the GPU at once):

```
block chunk size (1~8, recommended 4):
```

| CHUNK | Layers on GPU | With prefetch | Approx. VRAM |
|-------|---------------|---------------|--------------|
| 1     | 1             | 2             | ~0.5GB       |
| 2     | 2             | 4             | ~1.1GB       |
| 4     | 4             | 8             | ~2.2GB       |
| 8     | 8             | 16            | ~4.3GB       |

- Larger chunk = faster, more VRAM.
- Smaller chunk = safer on low VRAM.
- Up to `2 × CHUNK` layers are resident with prefetch.

Trade-off: VRAM drops from ~15GB to ~3GB, but inference is slower due to
CPU-GPU PCIe transfers. Output quality may also be slightly lower than full
GPU mode (differences in floating-point precision and kernel accumulation
order across CPU↔GPU transfers).