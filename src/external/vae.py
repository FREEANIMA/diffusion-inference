from diffusers import AutoencoderDC
from PIL import Image
import torch
import torch.nn.functional as F
from src.core import state

# FLUX	black-forest-labs/FLUX.1-dev (vae 서브폴더)	8×	16
# DC-AE 32×	mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers	32×	32
# DC-AE 64×	mit-han-lab/dc-ae-f64c128-in-1.0-diffusers	64×	128

class VAE():
    def __init__(self, model_name="mit-han-lab/dc-ae-f32c32-sana-1.0-diffusers", console=None):
        self.model_name = model_name
        self.vae = self.download_flux_vae()
        
        if console is not None:
            console.put("vae complete")

        self.vae.eval()

    def download_flux_vae(self):
        vae = AutoencoderDC.from_pretrained(
            self.model_name,
            torch_dtype=state.config.torch_dtype
        )
        vae.to(state.config.device)
        vae.eval()
        vae.requires_grad_(False)
        return vae

    @torch.inference_mode()
    def encoder(self, img, chunk_size=8):
        img = img.to(device=state.config.device, dtype=state.config.torch_dtype, non_blocking=True)

        if img.min() >= 0.0 and img.max() <= 1.0:
            img = img * 2.0 - 1.0

        latents = []
        for i in range(0, img.shape[0], chunk_size):
            chunk = img[i:i+chunk_size]
            latent = self.vae.encode(chunk).latent
            latents.append(latent)

        return torch.cat(latents, dim=0)

    @staticmethod
    def _tensor_to_pil(img_tensor: torch.Tensor, thumb_size: int = None) -> Image.Image:
        img = img_tensor.detach().float().cpu()

        if img.dim() == 4:        # (B, C, H, W) → (C, H, W)
            img = img[0]

        img = img.clamp(0.0, 1.0)
        img = (img * 255.0).round().to(torch.uint8)
        img = img.permute(1, 2, 0).numpy()

        if img.shape[-1] == 1:
            img = img.repeat(3, axis=-1)

        pil_img = Image.fromarray(img)
        if thumb_size is not None:
            pil_img = pil_img.resize((thumb_size, thumb_size), Image.BILINEAR)
        return pil_img

    @staticmethod
    def _tensor_to_pils(img_tensor: torch.Tensor) -> list[Image.Image]:
        img = img_tensor.detach().float().cpu()
        if img.dim() == 3:                # (C, H, W) → (1, C, H, W)
            img = img.unsqueeze(0)

        img = (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        img = img.permute(0, 2, 3, 1).numpy()      # (B, H, W, C)

        if img.shape[-1] == 1:
            img = img.repeat(3, axis=-1)

        return [Image.fromarray(a) for a in img]

    @torch.inference_mode()
    def decoder(self, latent, chunk_size=8) -> torch.Tensor:
        latent = latent.to(device=state.config.device, dtype=state.config.torch_dtype, non_blocking=True)
        
        images = []
        for i in range(0, latent.shape[0], chunk_size):
            chunk = latent[i:i+chunk_size]
            decoded = self.vae.decode(chunk).sample
            images.append(decoded)

        images = torch.cat(images, dim=0)

        # [-1, 1] -> [0, 1]
        images = (images * 0.5 + 0.5).clamp(0.0, 1.0)

        return images
    
    def to(self, device):
        self.vae.to(device)
        state.config.device = str(device)
        return self    