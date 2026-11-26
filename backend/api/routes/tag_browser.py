from fastapi import APIRouter, Request

router = APIRouter(prefix="/api", tags=["tags"])

@router.get("/tags")
def get_tags(request: Request):
    return request.app.state.tags