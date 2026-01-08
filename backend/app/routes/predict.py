from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..state import AppState
from pathlib import Path
import joblib

"""Example instance:
{
  "FlightDate": "04/04/2022",
  "CRSDepTime": 1133,
  "CRSElapsedTime": 72.0,
  "Distance": 212.0,
  "Quarter": 2,
  "Month": 4,
  "DayofMonth": 4,
  "DayOfWeek": 1,
  "Marketing_Airline_Network": "UA",
  "Operated_or_Branded_Code_Share_Partners": "UA_CODESHARE",
  "DOT_ID_Marketing_Airline": 19977,
  "Flight_Number_Marketing_Airline": 4301,
  "Operating_Airline": "C5",
  "Tail_Number": "N21144",
  "Flight_Number_Operating_Airline": 4301,
  "OriginAirportID": 11921,
  "OriginCityMarketID": 31921,
  "OriginState": "CO",
  "DestAirportID": 11292,
  "DestCityMarketID": 30325,
  "DestState": "CO",
  "DepTimeBlk": "1100-1159",
  "CRSArrTime": 1245,
  "ArrTimeBlk": "1200-1259",
  "DistanceGroup": 1
}
"""

router = APIRouter()


def get_state() -> AppState:
    from ..main import state  # type: ignore
    return state


class PredictRequest(BaseModel):
    model_id: str
    instance: Dict[str, Any]


def to_python(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

@router.post("/predict-single")
def predict_single(req: PredictRequest, state: AppState = Depends(get_state)):
    if req.model_id not in state.models:
        raise HTTPException(status_code=400, detail=f"Unknown model_id: {req.model_id}")

    model = state.models[req.model_id]

    # Build a one-row DataFrame
    X = pd.DataFrame([req.instance])

    # Only apply encoding and scaling steps at inference
    def _resolve_config_dir() -> Path:
        root_config = Path(__file__).resolve().parents[2] / "prediction_objects.json"
        backend_config = Path(__file__).resolve().parents[1] / "prediction_objects.json"
        cfg = root_config if root_config.exists() else backend_config
        return cfg.parent

    steps: List[str] = state.metadata.get("pipeline_steps", []) if state and state.metadata else []
    enc_scale_steps: List[str] = []
    if steps:
        for s in steps:
            ls = s.lower()
            if ("encoder" in ls) or ("scaling" in ls):
                enc_scale_steps.append(s)

    # If metadata not provided, default to encoder+scaling in pipeline folder
    if not enc_scale_steps:
        enc_scale_steps = [
            "pipeline/encoder.joblib",
            "pipeline/scaling.joblib",
        ]

    base_dir = _resolve_config_dir()
    X_transformed = X
    for step_path in enc_scale_steps:
        sp = Path(step_path)
        if not sp.is_absolute():
            sp = base_dir / step_path
        if not sp.exists():
            raise HTTPException(status_code=500, detail=f"Inference step not found: {sp}")
        try:
            transformer = joblib.load(sp)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load inference step {sp.name}: {e}")

        try:
            X_transformed = transformer.transform(X_transformed)
        except AttributeError:
            X_transformed = transformer(X_transformed)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference step {sp.name} failed: {e}")

    # Predict
    try:
        y_pred = model.predict(X_transformed)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    pred_py = to_python(y_pred[0])
    if pred_py == 1:
        pred_class = "Cancelled"
    else:
        pred_class = "Not Cancelled"

    # Optional: probability if available
    proba_py = None
    if hasattr(model, "predict_proba"):
        try:
            p = model.predict_proba(X_transformed)
            proba_py = to_python(p[0][pred_py])
            proba_py = float(p[0][pred_py]) * 100
            proba_py = round(proba_py, 1)
        except Exception:
            proba_py = None

    return {"model_id": req.model_id, "prediction": pred_class, "proba": proba_py}
