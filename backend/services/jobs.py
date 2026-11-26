import threading
import uuid
from typing import Callable

from fastapi import HTTPException

from services import pil_to_b64
jobs: dict[str, dict] = {}
cancels: dict[str, threading.Event] = {}
gpu_lock = threading.Lock()


class Cancelled(BaseException):
    """정지 요청. BaseException이라 engine의 `except Exception`(에러 로그)에 안 걸리고 그대로 빠져나옴."""


def start_job(total: int, fn: Callable, preview_size: int = 512) -> str:
    """fn(on_step, on_preview) -> list[PIL.Image] 를 백그라운드 스레드에서 실행하고 job_id 반환."""
    job_id = uuid.uuid4().hex
    cancel = threading.Event()
    cancels[job_id] = cancel
    jobs[job_id] = {"status": "running", "step": 0, "total": total,
                    "images": None, "error": None,
                    "preview": None, "preview_seq": 0}

    def check():
        if cancel.is_set():
            raise Cancelled()

    def on_step(i, n):
        jobs[job_id].update(step=i, total=n)
        check()                                  # 스텝이 끝날 때마다 정지 확인

    def on_preview(i, n, im):
        check()
        im.thumbnail((preview_size, preview_size))
        jobs[job_id]["preview"] = pil_to_b64(im)
        jobs[job_id]["preview_seq"] += 1

    def work():
        try:
            with gpu_lock:
                check()                          # 대기열에서 기다리는 동안 정지했다면 바로 종료
                imgs = fn(on_step, on_preview)
            jobs[job_id].update(status="done", images=[pil_to_b64(im) for im in imgs])
        except Cancelled:
            jobs[job_id].update(status="cancelled")
        except Exception as e:
            jobs[job_id].update(status="error", error=str(e))
        finally:
            cancels.pop(job_id, None)

    threading.Thread(target=work, daemon=True).start()
    return job_id


def cancel_job(job_id: str) -> dict:
    cancel = cancels.get(job_id)
    if cancel is None:                           # 이미 끝났거나 없는 job
        raise HTTPException(status_code=404, detail="job not found")
    cancel.set()
    return {"ok": True}


def get_job(job_id: str, seen: int = 0) -> dict:
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    out = {k: v for k, v in job.items() if k != "preview"}
    if job["preview_seq"] > seen:
        out["preview"] = job["preview"]
    if job["status"] != "running":
        jobs.pop(job_id, None)
    return out