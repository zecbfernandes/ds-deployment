from fastapi import APIRouter
from fastapi import Depends
from ..state import AppState


router = APIRouter()


def get_state() -> AppState:
    # Import the singleton created in main
    from ..main import state  # type: ignore
    return state


@router.get("/models")
def list_models(state: AppState = Depends(get_state)):
    items = []
    for m in state.metadata.get("models", []):
        items.append({
            "id": m.get("id"),
            "name": m.get("name"),
            "type": m.get("type"),
        })
    return {"models": items, "has_pipeline": True}
