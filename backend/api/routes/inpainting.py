from fastapi import APIRouter, Depends, Request

from api.schemas import InPaintingRequest, SamplingRequest
from src.sampling import Engine, Txt2ImgInput, InpaintInput
from services import b64_to_image, b64_to_mask
from services.jobs import start_job, cancel_job, get_job

router = APIRouter(prefix="/api", tags=["inpainting"])


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


@router.post("/inpainting")
def start_sampling(sampling: SamplingRequest, inpainting: InPaintingRequest,
                   engine: Engine = Depends(get_engine)) -> dict:
    txt2img_input = Txt2ImgInput(
        prompts=sampling.prompts,
        steps=sampling.steps,
        seed=sampling.seed,
        calculate_type=sampling.calculate_type,
        sigma_curve=sampling.sigma_curve,
        cfg_curve=sampling.cfg_curve,
        H=sampling.height,
        W=sampling.width,
        eta_curve=sampling.eta_curve,
        frame_save=sampling.frame_save,
        prompt_static=sampling.prompt_static,
        nsfw=sampling.nsfw,
    )

    use_crop = inpainting.crop_b64 is not None and inpainting.crop_box is not None
    img = b64_to_image(inpainting.image_b64, "RGB")

    inpaint_input = InpaintInput(
        img=img,
        start_t=inpainting.start_t,
        mask=b64_to_mask(inpainting.mask_b64, img.size),
        crop_img=b64_to_image(inpainting.crop_b64, "RGB") if use_crop else None,
        crop_box=inpainting.crop_box if use_crop else None,
        upscale=use_crop,
    )

    def run(on_step, on_preview):
        # engine.inpaint 는 (imgs, frames, sigmas) 3개를 반환
        return engine.inpaint(txt2img_input=txt2img_input, inpaint_input=inpaint_input,
                              on_step=on_step, on_preview=on_preview)[0]

    return {"job_id": start_job(sampling.steps, run, preview_size=768)}


@router.post("/inpainting/{job_id}/cancel")
def cancel_sampling(job_id: str) -> dict:
    return cancel_job(job_id)


@router.get("/inpainting/{job_id}")
def get_sampling(job_id: str, seen: int = 0) -> dict:
    return get_job(job_id, seen)