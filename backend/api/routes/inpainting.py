from fastapi import APIRouter, Depends, Request, Response

from api.schemas import InPaintingRequest, SamplingRequest
from src.sampling import Engine, Txt2ImgInput, InpaintInput
from services import b64_to_image

router = APIRouter(prefix="/api", tags=["inpainting"])

def get_engine(request: Request) -> Engine:
    return request.app.state.engine

@router.post("/inpainting", status_code=204)
def run_inpainting(sampling: SamplingRequest, inpainting: InPaintingRequest, engine: Engine = Depends(get_engine)) -> Response:
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
        nsfw=sampling.nsfw
    )

    use_crop = inpainting.crop_b64 is not None and inpainting.crop_box is not None

    inpaint_input = InpaintInput(
        img=b64_to_image(inpainting.image_b64, "RGB"),
        start_t=inpainting.start_t,
        mask=b64_to_image(inpainting.mask_b64, "L"),
        crop_img=b64_to_image(inpainting.crop_b64, "RGB") if use_crop else None,
        crop_box=inpainting.crop_box if use_crop else None,
        upscale=use_crop,
    )
    engine.inpaint(txt2img_input=txt2img_input, inpaint_input=inpaint_input)
    
    return Response(status_code=204)