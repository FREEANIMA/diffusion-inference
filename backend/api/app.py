from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from contextlib import asynccontextmanager
from src.sampling import Engine

from services import load_tags
from api.routes import inpainting, sampling, tag_browser

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.engine = Engine(console=None)
    app.state.tags = load_tags()
    yield


class SPAStaticFiles(StaticFiles):
    """파일이 없는 경로(/sampling, /inpainting)는 index.html을 돌려줘 JS 라우터가 처리하게 함"""
    async def get_response(self, path, scope):
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as e:
            is_file = "." in Path(path).name          # .css, .js 등 파일 요청은 404 그대로
            if e.status_code == 404 and not is_file and not path.startswith("api"):
                return await super().get_response("index.html", scope)
            raise

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/sampling")

@app.middleware("http")
async def no_cache(request, call_next):
    resp = await call_next(request)
    if not request.url.path.startswith("/api"):
        resp.headers["Cache-Control"] = "no-store"
    return resp

app.include_router(sampling.router)
app.include_router(inpainting.router)
app.include_router(tag_browser.router)

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/", SPAStaticFiles(directory=FRONTEND, html=True), name="frontend")