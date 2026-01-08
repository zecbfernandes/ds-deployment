from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from ..state import AppState
from sklearn.metrics import accuracy_score, f1_score
from pathlib import Path
import logging
import joblib


router = APIRouter()
logger = logging.getLogger("uvicorn")


def get_state() -> AppState:
    from ..main import state  # type: ignore
    return state


@router.post("/evaluate-models")
async def evaluate_models(
    file: UploadFile = File(...),
    target_column: Optional[str] = Form(None),
    state: AppState = Depends(get_state),
):
    if file.content_type not in {"text/csv", "application/vnd.ms-excel", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a CSV.")

    content = await file.read()
    # Read CSV with delimiter auto-detection; fallback to ';' and ','
    try:
        df = pd.read_csv(pd.io.common.BytesIO(content), sep=None, engine="python")
    except Exception:
        try:
            df = pd.read_csv(pd.io.common.BytesIO(content), sep=";")
        except Exception:
            try:
                df = pd.read_csv(pd.io.common.BytesIO(content), sep=",")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to read CSV: {e}")

    # Normalize column names to avoid whitespace mismatches
    try:
        df.columns = df.columns.str.strip()
    except Exception:
        pass

    # Determine target column with clearer error messages
    metadata_target = state.metadata.get("target_column") if state and state.metadata else None
    tgt_col = target_column or metadata_target
    # Debug: log all columns in the uploaded dataset
    available_columns = [str(c) for c in df.columns.tolist()]
    logger.info("Uploaded dataset columns: %s", available_columns)

    if tgt_col is None:
        raise HTTPException(
            status_code=400,
            detail=(
                "Target column missing: provide 'target_column' in the form data "
                "or configure it in the app metadata. "
                f"Available columns: {', '.join(map(str, df.columns.tolist()))}"
            ),
        )

    if tgt_col not in df.columns:
        source = "request" if target_column else "metadata"
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid target column '{tgt_col}' ({source}): not found in dataset columns. "
                f"Available columns: {', '.join(map(str, df.columns.tolist()))}"
            ),
        )

    

    # Keep the target column in the data; we'll separate at the end
    y = df[tgt_col]
    X = df

    # Log feature columns before any transformation
    try:
        feature_cols = [str(c) for c in X.columns.tolist()]
        logger.info("Feature columns before pipeline: %s", feature_cols)
    except Exception:
        logger.info("Feature data type before pipeline: %s", type(X))

    # Helper to resolve config directory (same logic as app startup)
    def _resolve_config_dir() -> Path:
        root_config = Path(__file__).resolve().parents[2] / "prediction_objects.json"
        backend_config = Path(__file__).resolve().parents[1] / "prediction_objects.json"
        cfg = root_config if root_config.exists() else backend_config
        return cfg.parent

    # Helper to stringify a small snapshot of the dataset safely
    def _snapshot(data: Any, rows: int = 5) -> str:
        try:
            # Prefer a concise JSON of the first few rows for DataFrames/Series
            if isinstance(data, pd.DataFrame):
                return data.head(rows).to_json(orient="records")
            if isinstance(data, pd.Series):
                return data.head(rows).to_json(orient="records")
        except Exception:
            pass
        # Fallback for numpy arrays or other matrix-like objects
        try:
            shape = getattr(data, "shape", None)
            sample = None
            try:
                sample_slice = data[:rows]
                sample = (
                    sample_slice.toarray().tolist() if hasattr(sample_slice, "toarray")
                    else sample_slice.tolist() if hasattr(sample_slice, "tolist")
                    else str(sample_slice)
                )
            except Exception:
                sample = None
            return f"type={type(data).__name__}, shape={shape}, sample={sample}"
        except Exception as e:
            return f"type={type(data).__name__}, error={e}"

    # Load and apply full pipeline if steps provided in metadata
    steps: List[str] = state.metadata.get("pipeline_steps", []) if state and state.metadata else []
    if steps:
        base_dir = _resolve_config_dir()

        X_transformed = X
        # Temporarily skip selected pipeline steps
        SKIP_STEPS = {"balancer.joblib", "mv.joblib", "outlier_processor.joblib"}
        # Print dataset before the first step
        try:
            logger.info("Dataset BEFORE first pipeline step (sample): %s", _snapshot(X_transformed))
        except Exception:
            logger.info("Unable to snapshot dataset before steps", exc_info=True)
        for step_path in steps:
            sp = Path(step_path)
            if not sp.is_absolute():
                sp = base_dir / step_path
            # Skip configured steps by filename
            if sp.name in SKIP_STEPS:
                logger.info("Skipping pipeline step: %s", sp.name)
                continue
            if not sp.exists():
                raise HTTPException(status_code=500, detail=f"Pipeline step not found: {sp}")
            try:
                transformer = joblib.load(sp)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to load pipeline step {sp.name}: {e}")

            try:
                # Log input columns for this step
                try:
                    step_input_cols = [str(c) for c in getattr(X_transformed, "columns", []).tolist()]
                    logger.info("Step %s input columns: %s", sp.name, step_input_cols)
                except Exception:
                    logger.info("Step %s input type: %s", sp.name, type(X_transformed))
                # Print dataset snapshot before applying this step
                try:
                    logger.info("Dataset BEFORE '%s' (sample): %s", sp.name, _snapshot(X_transformed))
                    logger.info("\n")
                except Exception:
                    logger.info("Unable to snapshot dataset before '%s'", sp.name, exc_info=True)
                X_transformed = transformer.transform(X_transformed)
                # Log output columns after this step
                try:
                    step_output_cols = [str(c) for c in getattr(X_transformed, "columns", []).tolist()]
                    logger.info("Step %s output columns: %s", sp.name, step_output_cols)
                except Exception:
                    shape = getattr(X_transformed, "shape", None)
                    if shape is not None:
                        logger.info("Step %s output type: %s, shape=%s", sp.name, type(X_transformed), shape)
                    else:
                        logger.info("Step %s output type: %s", sp.name, type(X_transformed))
                # Print dataset snapshot after applying this step
                try:
                    logger.info("Dataset AFTER '%s' (sample): %s", sp.name, _snapshot(X_transformed))
                    logger.info("\n")
                except Exception:
                    logger.info("Unable to snapshot dataset after '%s'", sp.name, exc_info=True)
            except AttributeError:
                # Some steps may be callable
                X_transformed = transformer(X_transformed)
                # Log output columns after callable step
                try:
                    step_output_cols = [str(c) for c in getattr(X_transformed, "columns", []).tolist()]
                    logger.info("Step %s output columns: %s", sp.name, step_output_cols)
                except Exception:
                    shape = getattr(X_transformed, "shape", None)
                    if shape is not None:
                        logger.info("Step %s output type: %s, shape=%s", sp.name, type(X_transformed), shape)
                    else:
                        logger.info("Step %s output type: %s", sp.name, type(X_transformed))
                # Print dataset snapshot after callable step
                try:
                    logger.info("Dataset AFTER callable '%s' (sample): %s", sp.name, _snapshot(X_transformed))
                except Exception:
                    logger.info("Unable to snapshot dataset after callable '%s'", sp.name, exc_info=True)
            except Exception as e:
                present_cols: List[str] = []
                try:
                    present_cols = [str(c) for c in getattr(X_transformed, "columns", []).tolist()]
                except Exception:
                    pass
                raise HTTPException(
                    status_code=500,
                    detail=f"Pipeline step {sp.name} failed: {e}. Present columns: {present_cols}",
                )
    else:
        # Fallback to single pipeline object if configured
        if state.pipeline is not None:
            try:
                X_transformed = state.pipeline.transform(X)
            except AttributeError:
                X_transformed = state.pipeline(X)
            # Log columns after applying single pipeline
            try:
                pipeline_output_cols = [str(c) for c in getattr(X_transformed, "columns", []).tolist()]
                logger.info("Columns after pipeline: %s", pipeline_output_cols)
            except Exception:
                shape = getattr(X_transformed, "shape", None)
                if shape is not None:
                    logger.info("Data after pipeline type: %s, shape=%s", type(X_transformed), shape)
                else:
                    logger.info("Data after pipeline type: %s", type(X_transformed))
        else:
            X_transformed = X

    # Separate target/features at the end, if present in transformed data
    y_final = y
    X_final = X_transformed
    try:
        if hasattr(X_transformed, "columns") and tgt_col in X_transformed.columns:
            y_final = X_transformed[tgt_col]
            X_final = X_transformed.drop(columns=[tgt_col])
            try:
                final_feature_cols = [str(c) for c in getattr(X_final, "columns", []).tolist()]
                logger.info("Final feature columns used for prediction: %s", final_feature_cols)
            except Exception:
                shape = getattr(X_final, "shape", None)
                if shape is not None:
                    logger.info("Final feature data type: %s, shape=%s", type(X_final), shape)
                else:
                    logger.info("Final feature data type: %s", type(X_final))
    except Exception:
        # If transformed output isn't a DataFrame, fall back to original y
        pass

    results: List[Dict[str, Any]] = []
    for mid, model in state.models.items():
        try:
            y_pred = model.predict(X_final)
        except Exception as e:
            results.append({"model_id": mid, "error": f"Prediction failed: {e}"})
            continue

        acc = accuracy_score(y_final, y_pred)
        f1 = f1_score(y_final, y_pred, average="weighted")

        results.append({
            "model_id": mid,
            "metrics": {
                "accuracy": acc,
                "f1_weighted": f1,
            },
        })

    return {"results": results}
