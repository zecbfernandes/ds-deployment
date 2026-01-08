from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from .routes import models, predict, evaluate
from .state import load_prediction_objects, AppState

app = FastAPI()

# Frontend directory from repo root
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR), html=False), name="static")

# Application state shared across routers
state = AppState()


@app.on_event("startup")
def startup_event():
    # Try repo root first, then backend folder as fallback
    root_config = Path(__file__).resolve().parents[2] / "prediction_objects.json"
    backend_config = Path(__file__).resolve().parents[1] / "prediction_objects.json"
    config_path = root_config if root_config.exists() else backend_config
    load_prediction_objects(config_path, state)


# Serve SPA index at root without intercepting /api
@app.get("/")
def serve_index():
    index_file = FRONTEND_DIR / "index.html"
    if not index_file.exists():
        return {"error": "Frontend index.html not found"}
    from fastapi.responses import FileResponse
    return FileResponse(str(index_file))


app.include_router(models.router, prefix="/api", tags=["models"], dependencies=[], responses={})
app.include_router(predict.router, prefix="/api", tags=["predict"], dependencies=[], responses={})
app.include_router(evaluate.router, prefix="/api", tags=["evaluate"], dependencies=[], responses={})
