import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.app:app",
        app_dir="backend",
        host="0.0.0.0",
        port=8000,
        reload=False,
        access_log=False,
    )