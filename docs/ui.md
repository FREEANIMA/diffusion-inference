## UI

## UI Update

- **Multi-prompt** support
- **Custom schedulers**
  - Sigma scheduler (CDF-based)
  - CFG scheduler
  - Eta scheduler
- **Click-button revision**

## Screenshots

### Sampling
![Sampling UI](docs/imgs/new_ui.png)

### Inpainting
![Inpainting UI](docs/imgs/new_ui_2.png)

---

- Three tabs: **Text2img**, **Inpainting**, **Detailer**. The last tab is remembered, and `#txt2img`, `#inpainting`, `#detailer` in the URL open that tab directly.
- **Tag Browser**: browse and search tags by type, and click to add them to the prompt.
- **Tags autocomplete** and **History**: click a history entry to restore the input plus the expanded tags.
- **고정 (Fixed)** switch: when on, only the entered tags are used and no LLM tag expansion happens.
- Only one job runs at a time. Generation runs in a background thread and the browser polls for the result, so long jobs do not hit proxy timeouts.

### UI
## Detailing & Inpainting

Both tabs share the same mask painter with two layers.

| Color | Mode | Meaning |
|-------|------|---------|
| Red | 재생성 (Regen) | Area to redraw |
| Blue | 보존 (Keep) | Area that never changes |
| – | 지우개 (Erase) | Remove paint from both layers |

- Painting one color erases the other under the brush. The later stroke wins, so the layers never overlap.
- If nothing is painted, the whole image (inpainting) or the whole crop (detailer) is redrawn.
- Keep overrides Regen. The final redraw area is `regen × (1 − keep)`.
- Masks are resized to the sampling resolution with nearest-neighbor, so they stay binary.
- **Start T** (0–1): the point in the schedule where redrawing starts. Higher values change more, lower values stay closer to the original. For a full redraw without a mask, around 0.6 is recommended.
- **Brush**: brush size in px.

---

### Inpainting

1. Enter the image path and press **Load**.
2. Paint the Regen / Keep areas on the Original panel.
3. Press **Generate**. The result appears on the right and is saved to `inpainting/`.

The image is fitted so that the long side is 1024 and the short side is snapped to a multiple of `patch_size × compression_ratio`.

### Detailer

1. Enter the image path and press **Load**.
2. Drag a rectangle on the Original panel. The rectangle snaps to multiples of `patch_size × compression_ratio`, and only sizes that scale cleanly to `allowable_size` on the long side are allowed.
3. The selected area appears enlarged in **Crop · Mask**. Paint the Regen / Keep areas there. The label under the crop shows the original size → the enlarged size.
4. Press **Generate**. The crop is redrawn at the enlarged size, scaled back down, and pasted into the original image. The result is saved to `detailer/`.

---
