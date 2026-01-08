from pathlib import Path
from typing import Any, Dict, List, Optional

import json
import joblib


class AppState:
    pipeline: Optional[Any]
    models: Dict[str, Any]
    metadata: Dict[str, Any]

    def __init__(self) -> None:
        self.pipeline = None
        self.models = {}
        self.metadata = {}


def load_prediction_objects(config_path: Path, state: AppState) -> None:
    if not config_path.exists():
        raise FileNotFoundError(f"Prediction config not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Load pipeline if provided
    pipeline_file = cfg.get("pipeline")
    if pipeline_file:
        pf = (config_path.parent / pipeline_file) if not pipeline_file.startswith("/") else Path(pipeline_file)
        if not pf.exists():
            raise FileNotFoundError(f"Pipeline file not found: {pf}")
        state.pipeline = joblib.load(pf)
    else:
        state.pipeline = None

    # Load models
    models_cfg: List[Dict[str, Any]] = cfg.get("models", [])
    for m in models_cfg:
        mid = m["id"]
        mfile = m["file"]
        mf = (config_path.parent / mfile) if not mfile.startswith("/") else Path(mfile)
        if not mf.exists():
            raise FileNotFoundError(f"Model file not found for '{mid}': {mf}")
        state.models[mid] = joblib.load(mf)

    state.metadata = cfg
