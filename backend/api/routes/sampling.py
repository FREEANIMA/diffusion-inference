import threading
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from api.schemas import SamplingRequest
from src.sampling import Engine, Txt2ImgInput
from services import pil_to_b64

router = APIRouter(prefix="/api", tags=["sampling"])

jobs: dict[str, dict] = {}
cancels: dict[str, threading.Event] = {}     # job_id -> 정지 신호 (jobs에 넣으면 JSON 직렬화가 깨져서 따로 보관)
gpu_lock = threading.Lock()


class Cancelled(BaseException):
    """정지 요청. BaseException이라 engine의 `except Exception`(에러 로그)에 안 걸리고 그대로 빠져나옴."""


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
    cancel = threading.Event()
    cancels[job_id] = cancel
    jobs[job_id] = {"status": "running", "step": 0, "total": sampling.steps,
                    "images": None, "error": None,
                    "preview": None, "preview_seq": 0}

    def check():
        if cancel.is_set():
            raise Cancelled()

    def work():
        def on_step(i, n):
            jobs[job_id].update(step=i, total=n)
            check()                                  # 스텝이 끝날 때마다 정지 확인

        def on_preview(i, n, im):
            check()
            im.thumbnail((512, 512))
            jobs[job_id]["preview"] = pil_to_b64(im)
            jobs[job_id]["preview_seq"] += 1

        try:
            with gpu_lock:
                check()                              # 대기열에서 기다리는 동안 정지했다면 바로 종료
                imgs, _, _ = engine.txt2img(txt2img_input=txt2img_input, inpaint_ctx=None,
                                            on_step=on_step, on_preview=on_preview)
            jobs[job_id].update(status="done", images=[pil_to_b64(im) for im in imgs])
        except Cancelled:
            jobs[job_id].update(status="cancelled")
        except Exception as e:
            jobs[job_id].update(status="error", error=str(e))
        finally:
            cancels.pop(job_id, None)

    threading.Thread(target=work, daemon=True).start()
    return {"job_id": job_id}


@router.post("/sampling/{job_id}/cancel")
def cancel_sampling(job_id: str) -> dict:
    cancel = cancels.get(job_id)
    if cancel is None:                               # 이미 끝났거나 없는 job
        raise HTTPException(status_code=404, detail="job not found")
    cancel.set()
    return {"ok": True}


@router.get("/sampling/{job_id}")
def get_sampling(job_id: str, seen: int = 0) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    out = {k: v for k, v in job.items() if k != "preview"}
    if job["preview_seq"] > seen:
        out["preview"] = job["preview"]
    if job["status"] != "running":
        jobs.pop(job_id, None)
    return out