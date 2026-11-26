import threading
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response

from api.schemas import InPaintingRequest, SamplingRequest
from src.sampling import Engine, Txt2ImgInput, InpaintInput
from services import b64_to_image, pil_to_b64

router = APIRouter(prefix="/api", tags=["inpainting"])

jobs: dict[str, dict] = {}
gpu_lock = threading.Lock()

def get_engine(request: Request) -> Engine:
    return request.app.state.engine

@router.post("/inpainting")
def start_sampling(sampling: SamplingRequest, inpainting: InPaintingRequest, engine: Engine = Depends(get_engine)) -> Response:
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

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "running", "step": 0, "total": sampling.steps,
                    "images": None, "error": None}

    def work():
        def on_step(i, n):
            jobs[job_id].update(step=i, total=n)
        try:
            with gpu_lock:
                imgs, _, _ = engine.inpaint(
                    txt2img_input=txt2img_input,   
                    inpaint_input=inpaint_input,            
                    on_step=on_step
                )
            jobs[job_id].update(status="done", images=[pil_to_b64(im) for im in imgs])
        except Exception as e:
            jobs[job_id].update(status="error", error=str(e))

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}

@router.get("/inpainting/{job_id}")
def get_sampling(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] != "running":
        jobs.pop(job_id, None)
    return job