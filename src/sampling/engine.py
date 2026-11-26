import torch
from PIL import Image
import numpy as np
from src.core import log, state, error_report, dbg
from src.external import TE, VAE
from src.network import MMDIT
from src.tag import TagGenration
from src.storage import download_model, load_ema_only
import os
import gc
from datetime import datetime

import numpy as np
from src.sampling.engine_internal import (
    Txt2ImgInput, InitOutput, InpaintInput, InpaintCtx,
    setup_init, sampling, arr_resize, mask_generation, save_img, save_video, save_attention_plot, save_attention_json
)

class SamplingEngine:
    def __init__(self, console=None):
        self.console = console

        # test
        self.model = None

        self.te = TE()
        self.vae = VAE()

        self.llm:TagGenration = TagGenration(console=console)

        download_model(config=state.config, filename=state.config.EMA_FILENAME)
        self.model:MMDIT = load_ema_only(
            config=state.config,
            network=MMDIT,
            console=console
        )
        

    @torch.inference_mode()
    def vae_decode(self, x: torch.Tensor, latents: list[torch.Tensor], frame_save: bool):
        imgs = self.vae._tensor_to_pils(self.vae.decoder(x))

        frames: list[list[Image.Image]] = [[] for _ in range(x.shape[0])]
        if frame_save and latents:
            for latent in latents:
                pils = self.vae._tensor_to_pils(self.vae.decoder(latent))
                for b, im in enumerate(pils):
                    frames[b].append(im)

        return imgs, frames


    @torch.inference_mode()
    def preview_decode(self, z: torch.Tensor) -> Image.Image:
        return self.vae._tensor_to_pils(self.vae.decoder(z))[0]


    def save(self, imgs, frames, txt2img_input, save_dir, layer_stats_steps=None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(save_dir, ts)
        os.makedirs(out_dir, exist_ok=True)

        for i, img in enumerate(imgs):
            save_img(os.path.join(out_dir, f"img_{i}.png"), img)

        if txt2img_input.frame_save:
            for b, frames_b in enumerate(frames):
                save_video(
                    os.path.join(out_dir, f"frames_{b}.mp4"),
                    frames_b,
                    fps=16
                )

        if layer_stats_steps:
            save_attention_plot(
                layer_stats_steps,
                os.path.join(out_dir, "attention_strength.png")
            )

            save_attention_json(
                layer_stats_steps,
                os.path.join(out_dir, "attention_strength.json")
            )


    
    def txt2img(self, txt2img_input:Txt2ImgInput, start_t=1.0, inpaint_ctx: InpaintCtx | None = None, on_step=None, on_preview=None, save_dir=state.config.GENERATION_DIR) -> tuple[list[Image.Image], list[Image.Image], list[float]]:
        try:
            txt2img_input.prompts = self.llm.generation(
                tags=txt2img_input.prompts, 
                nsfw=txt2img_input.nsfw,
                H=txt2img_input.H,
                W=txt2img_input.W,
                inferenc_mode=txt2img_input.prompt_static
            )
        
            init_out:InitOutput = setup_init(model=self.te, config=state.config, txt2img_input=txt2img_input)

            start_idx = int(np.argmin(np.abs(init_out.sigmas - start_t)))
            init_out.sigmas = init_out.sigmas[start_idx:]
            init_out.cfgs = init_out.cfgs[start_idx:]
            init_out.etas = init_out.etas[start_idx:]

            print("========init_out=============")
            dbg(init_out)
            print("========txt2img_input=============")
            dbg(txt2img_input)
            print("======== PROMPTS =========")
            for i, prompt in enumerate(txt2img_input.prompts, start=1):
                print(f'prompt_{i} = "{prompt}"')
            print("==========================")
            print("===========================")

            self.model.eval()
            def _on_preview(i, n, z):
                on_preview(i, n, self.preview_decode(z))

            x, latents, sigmas_steps, layer_stats_steps = sampling(
                model=self.model,
                txt2img_input=txt2img_input,
                init_out=init_out,
                inpaint_ctx=inpaint_ctx,
                on_step=on_step,
                on_preview=_on_preview if on_preview else None,
                preview_every=4,
            )

            imgs, frames = self.vae_decode(x, latents, txt2img_input.frame_save)

            self.save(
                imgs=imgs,
                frames=frames,
                txt2img_input=txt2img_input,
                save_dir=save_dir,
                layer_stats_steps=layer_stats_steps
            )
            return imgs, frames, sigmas_steps
        
        except Exception as e:
            error_report(e, "txt2img")
            log(self.console, e)
        finally:
            del txt2img_input, init_out, imgs, frames, sigmas_steps, x, latents, layer_stats_steps, inpaint_ctx
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            raise

    def inpaint(self, txt2img_input:Txt2ImgInput, inpaint_input:InpaintInput, on_step=None, on_preview=None) -> tuple[list[Image.Image], list[Image.Image], list[float]]:
        if inpaint_input.crop_img is not None:
            img = inpaint_input.crop_img
        else:
            img = inpaint_input.img

        origin_img = inpaint_input.img

        old_W, old_H = img.size
        img_side = max(old_W, old_H)

        try:
            arr, H, W = arr_resize(
                np.asarray(img.convert('RGB')),
                max(1024, img_side),
                state.config.patch_size,
                state.config.compression_ratio,
                upscale=inpaint_input.upscale
            )

            txt2img_input.H = H
            txt2img_input.W = W
     
            mask = inpaint_input.mask.convert("L")
            if inpaint_input.crop_img is not None:
                x, y, w, h = inpaint_input.crop_box
                mask = mask.crop((x, y, x + w, y + h))
                
            mask = mask.resize((W, H), Image.Resampling.NEAREST)
            mask_arr = np.array(mask, dtype=np.uint8)

            np_tensor = torch.from_numpy(arr).permute(2, 0, 1).float().div_(255.0).unsqueeze(0)
            np_tensor = np_tensor.to(device=state.config.device, dtype=state.config.torch_dtype)

            latent = self.vae.encoder(np_tensor)
            mask_lat = mask_generation(mask_arr, latent)

            B = len(txt2img_input.prompts)

            ctx = InpaintCtx(
                z0=latent.expand(B, -1, -1, -1),
                mask=mask_lat.expand(B, -1, -1, -1)
            )

            return self.txt2img(txt2img_input, start_t=inpaint_input.start_t, inpaint_ctx=ctx, on_step=on_step, on_preview=on_preview, save_dir=state.config.INPAINTING_DIR)
        
        except Exception as e:
            error_report(e, "inpaint")
            log(self.console, e)
            raise    
