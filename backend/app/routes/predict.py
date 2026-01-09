from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ..state import AppState
from pathlib import Path
import joblib
from scipy import sparse

"""Example instance:

Non-cancelled flight example:
{   "FlightDate": "04/04/2022",   "CRSDepTime": 1133,   "CRSElapsedTime": 72.0,   "Distance": 212.0,   "Quarter": 2,   "Month": 4,   "DayofMonth": 4,   "DayOfWeek": 1,   "Marketing_Airline_Network": "UA",   "Operated_or_Branded_Code_Share_Partners": "UA_CODESHARE",   "Operating_Airline": "C5",   "Tail_Number": "N21144",   "OriginAirportID": 11921,   "OriginCityMarketID": 31921,   "OriginState": "CO",   "DestAirportID": 11292,   "DestCityMarketID": 30325,   "DestState": "CO",   "DepTimeBlk": "1100-1159",   "CRSArrTime": 1245,   "ArrTimeBlk": "1200-1259",   "DistanceGroup": 1 }

Cancelled flight example:
{   "FlightDate": "2022-04-04",  "CRSDepTime": 1529,   "CRSElapsedTime": 70.0,   "Distance": 251.0,   "Quarter": 2,   "Month": 4,   "DayofMonth": 4,   "DayOfWeek": 1,   "Marketing_Airline_Network": "UA",   "Operated_or_Branded_Code_Share_Partners": "UA_CODESHARE",   "Operating_Airline": "C5",   "Tail_Number": "N21144",   "OriginAirportID": 11413,   "OriginCityMarketID": 30285,   "OriginState": "CO",   "DestAirportID": 11292,   "DestCityMarketID": 30325,   "DestState": "CO",   "DepTimeBlk": "1500-1559",   "CRSArrTime": 1639,   "ArrTimeBlk": "1600-1659",   "DistanceGroup": 2 }

Missing value example:
{   "CRSDepTime": 1529,   "CRSElapsedTime": 70.0,   "Distance": 251.0,   "Quarter": 2,   "Month": 4,   "DayofMonth": 4,   "DayOfWeek": 1,   "Marketing_Airline_Network": "UA",   "Operated_or_Branded_Code_Share_Partners": "UA_CODESHARE",   "Operating_Airline": "C5",   "Tail_Number": "N21144",   "OriginAirportID": 11413,   "OriginCityMarketID": 30285,   "OriginState": "CO",   "DestAirportID": 11292,   "DestCityMarketID": 30325,   "DestState": "CO",   "DepTimeBlk": "1500-1559",   "CRSArrTime": 1639,   "ArrTimeBlk": "1600-1659",   "DistanceGroup": 2 }

Value not proper format example:
{   "FlightDate": "2022-04-04",  "CRSDepTime": "twelve",   "CRSElapsedTime": 70.0,   "Distance": 251.0,   "Quarter": 2,   "Month": 4,   "DayofMonth": 4,   "DayOfWeek": 1,   "Marketing_Airline_Network": "UA",   "Operated_or_Branded_Code_Share_Partners": "UA_CODESHARE",   "Operating_Airline": "C5",   "Tail_Number": "N21144",   "OriginAirportID": 11413,   "OriginCityMarketID": 30285,   "OriginState": "CO",   "DestAirportID": 11292,   "DestCityMarketID": 30325,   "DestState": "CO",   "DepTimeBlk": "1500-1559",   "CRSArrTime": 1639,   "ArrTimeBlk": "1600-1659",   "DistanceGroup": 2 }

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

    # Detect missing/blank inputs in the raw instance (including strings with only whitespace)
    def _is_missing(val: Any) -> bool:
        try:
            if isinstance(val, str):
                return val.strip() == ""
            return pd.isna(val)
        except Exception:
            return False

    raw_missing_fields: List[str] = []
    try:
        raw_missing_fields = [col for col, val in X.iloc[0].items() if _is_missing(val)]
    except Exception:
        raw_missing_fields = []

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
            # Keep track of input columns to attempt feature name preservation
            input_cols = list(getattr(X_transformed, "columns", []))
            X_transformed = transformer.transform(X_transformed)
            # Try to preserve/assign feature names after transform for better diagnostics
            try:
                feature_names = None
                if hasattr(transformer, "get_feature_names_out"):
                    try:
                        feature_names = transformer.get_feature_names_out(input_cols)
                    except Exception:
                        feature_names = transformer.get_feature_names_out()
                # If output is sparse, convert to dense for name assignment
                if sparse.issparse(X_transformed):
                    X_transformed = X_transformed.toarray()
                if isinstance(X_transformed, np.ndarray):
                    if feature_names is not None:
                        X_transformed = pd.DataFrame(X_transformed, columns=[str(n) for n in feature_names])
                    elif input_cols and X_transformed.ndim == 2 and X_transformed.shape[1] == len(input_cols):
                        X_transformed = pd.DataFrame(X_transformed, columns=input_cols)
            except Exception:
                # Best-effort naming; continue without names if unavailable
                pass
        except AttributeError:
            X_transformed = transformer(X_transformed)
            # If callable returns ndarray, wrap with previous column names when feasible
            try:
                prev_cols = list(getattr(X, "columns", []))
                if isinstance(X_transformed, np.ndarray) and prev_cols and X_transformed.shape[1] == len(prev_cols):
                    X_transformed = pd.DataFrame(X_transformed, columns=prev_cols)
            except Exception:
                pass
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Inference step {sp.name} failed: {e}")

    # Predict
    try:
        y_pred = model.predict(X_transformed)
    except Exception as e:
        # Collect detailed NaN diagnostics for user-friendly feedback
        transformed_nan_cols: List[str] = []
        transformed_nan_indices: List[int] = []
        try:
            if hasattr(X_transformed, "isna"):
                # DataFrame path
                nan_mask = X_transformed.isna()
                transformed_nan_cols = [c for c in X_transformed.columns.tolist() if nan_mask[c].any()]
            else:
                # ndarray/sparse path
                arr = X_transformed
                if sparse.issparse(arr):
                    arr = arr.toarray()
                if isinstance(arr, np.ndarray):
                    nan_mask = np.isnan(arr)
                    if arr.ndim == 2:
                        transformed_nan_indices = sorted(set(np.where(nan_mask)[1].tolist()))
                    else:
                        transformed_nan_indices = sorted(set(np.where(nan_mask)[0].tolist()))
        except Exception:
            transformed_nan_cols = []
            transformed_nan_indices = []

        extra_info = []
        if raw_missing_fields:
            extra_info.append(f"missing input fields: {raw_missing_fields}")
        if transformed_nan_cols:
            extra_info.append(f"NaN after preprocessing (feature names): {transformed_nan_cols}")
        elif transformed_nan_indices:
            extra_info.append(f"NaN after preprocessing (feature indices): {transformed_nan_indices}")

        hint = (
            "Consider imputing or providing values; see scikit-learn impute docs: "
            "https://scikit-learn.org/stable/modules/impute.html"
        )
        detail_msg = f"Prediction failed: {e}. "
        if extra_info:
            detail_msg += "Details: " + "; ".join(extra_info) + ". "

        raise HTTPException(status_code=500, detail=detail_msg)

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
