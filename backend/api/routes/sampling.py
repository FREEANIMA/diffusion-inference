from fastapi import APIRouter, Depends, Request

from api.schemas import SamplingRequest
from src.sampling import Engine, Txt2ImgInput
from services.jobs import start_job, cancel_job, get_job

router = APIRouter(prefix="/api", tags=["sampling"])


def get_engine(request: Request) -> Engine:
    return request.app.state.engine


@router.post("/sampling")
def start_sampling(sampling: SamplingRequest, engine: Engine = Depends(get_engine)) -> dict:
    txt2img_input = Txt2ImgInput(
        prompts=sampling.prompts,
        steps=sampling.steps,
        seed=sampling.seed,
        H=sampling.height,
        W=sampling.width,
        calculate_type=sampling.calculate_type,
        sigma_curve=sampling.sigma_curve,
        cfg_curve=sampling.cfg_curve,
        eta_curve=sampling.eta_curve,
        frame_save=sampling.frame_save,
        prompt_static=sampling.prompt_static,
        nsfw=sampling.nsfw,
    )

    def run(on_step, on_preview):
        return engine.txt2img(txt2img_input=txt2img_input, inpaint_ctx=None,
                              on_step=on_step, on_preview=on_preview)[0]

    return {"job_id": start_job(sampling.steps, run, preview_size=512)}


@router.post("/sampling/{job_id}/cancel")
def cancel_sampling(job_id: str) -> dict:
    return cancel_job(job_id)


@router.get("/sampling/{job_id}")
def get_sampling(job_id: str, seen: int = 0) -> dict:
    return get_job(job_id, seen)