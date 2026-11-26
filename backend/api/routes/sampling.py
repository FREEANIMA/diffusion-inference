import threading
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from api.schemas import SamplingRequest
from src.sampling import Engine, Txt2ImgInput
from services import pil_to_b64

router = APIRouter(prefix="/api", tags=["sampling"])

jobs: dict[str, dict] = {}
gpu_lock = threading.Lock()   # GPU를 한 번에 한 작업만 쓰게


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

    job_id = uuid.uuid4().hex
    jobs[job_id] = {"status": "running", "step": 0, "total": sampling.steps,
                    "images": None, "error": None}

    def work():
        def on_step(i, n):
            jobs[job_id].update(step=i, total=n)
        try:
            with gpu_lock:
                imgs, _, _ = engine.txt2img(txt2img_input=txt2img_input,
                                            inpaint_ctx=None, on_step=on_step)
            jobs[job_id].update(status="done", images=[pil_to_b64(im) for im in imgs])
        except Exception as e:
            jobs[job_id].update(status="error", error=str(e))

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@router.get("/sampling/{job_id}")
def get_sampling(job_id: str) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    if job["status"] != "running":
        jobs.pop(job_id, None)   # 결과를 한 번 가져가면 정리
    return job