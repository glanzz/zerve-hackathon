"""
src/api/routes.py — APIRouter with all prediction and utility endpoints.

Routes:
  GET  /health          -> HealthResponse
  POST /predict         -> PredictionResponse (single user)
  POST /predict/batch   -> BatchPredictResponse (up to 1000 users)
  GET  /model-info      -> ModelInfoResponse (full metadata JSON)
  GET  /segments        -> stub (Mission B in progress)
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from fastapi import APIRouter, HTTPException

from src.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    UserFeatureInput,
)
from src.models.mission_a_predict import load_model, predict_batch, predict_single

router = APIRouter()

# ── Cached model loader ────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_model():
    """Load model once on first call; cached for the process lifetime."""
    return load_model()


def _get_metadata() -> Dict[str, Any]:
    """Read model_metadata.json from disk (small file, cheap)."""
    from config.config import CFG
    _meta_path = Path(CFG.MODEL_DIR) / "model_metadata.json"
    if not _meta_path.exists():
        raise FileNotFoundError(f"model_metadata.json not found at {_meta_path}")
    with open(_meta_path) as _f:
        return json.load(_f)


def _feature_input_to_dict(user: UserFeatureInput) -> Dict[str, float]:
    return user.model_dump()


# ── Routes ─────────────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, tags=["Utility"])
def health_check() -> HealthResponse:
    """Liveness + model-loaded check."""
    try:
        _pipeline, _feature_names, _meta = _get_model()
        return HealthResponse(
            status="ok",
            model_loaded=True,
            oof_auc=_meta.get("oof_auc"),
            oof_pr_auc=_meta.get("oof_pr_auc"),
            n_features=_meta.get("n_features"),
            training_samples=_meta.get("n_train"),
        )
    except Exception as exc:
        return HealthResponse(status=f"degraded: {exc}", model_loaded=False)


@router.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_single_user(
    user: UserFeatureInput,
    person_id: Optional[str] = None,
) -> PredictionResponse:
    """Predict upgrade probability for a single user."""
    _t0 = time.perf_counter()
    _pipeline, _feature_names, _meta = _get_model()
    _features = _feature_input_to_dict(user)

    _result = predict_single(_features)
    _elapsed_ms = (time.perf_counter() - _t0) * 1000

    return PredictionResponse(
        person_id=person_id,
        upgrade_probability=_result["upgrade_probability"],
        upgrade_risk_tier=_result["upgrade_risk_tier"],
        recommended_action=_result["recommended_action"],
        model_version=_meta.get("trained_at", "unknown"),
        inference_time_ms=round(_elapsed_ms, 3),
        top_3_features=_result.get("top_3_features", {}),
    )


@router.post("/predict/batch", response_model=BatchPredictResponse, tags=["Prediction"])
def predict_batch_users(body: BatchPredictRequest) -> BatchPredictResponse:
    """Vectorized prediction for up to 1000 users."""
    _t0 = time.perf_counter()
    _pipeline, _feature_names, _meta = _get_model()

    _rows = [_feature_input_to_dict(u) for u in body.users]
    _df = pd.DataFrame(_rows)

    _out_df = predict_batch(_df)
    _elapsed_ms = (time.perf_counter() - _t0) * 1000
    _per_user_ms = _elapsed_ms / max(len(_rows), 1)

    _predictions: List[PredictionResponse] = []
    for _i, _row in _out_df.iterrows():
        _predictions.append(PredictionResponse(
            person_id=None,
            upgrade_probability=float(_row["upgrade_probability"]),
            upgrade_risk_tier=str(_row["upgrade_risk_tier"]),
            recommended_action=str(_row["recommended_action"]),
            model_version=_meta.get("trained_at", "unknown"),
            inference_time_ms=round(_per_user_ms, 3),
            top_3_features={},
        ))

    return BatchPredictResponse(
        predictions=_predictions,
        n_users=len(_predictions),
        inference_time_ms=round(_elapsed_ms, 3),
    )


@router.get("/model-info", response_model=ModelInfoResponse, tags=["Utility"])
def model_info() -> ModelInfoResponse:
    """Return full model_metadata.json."""
    try:
        _meta = _get_metadata()
        return ModelInfoResponse(metadata=_meta)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/segments", tags=["Segments"])
def segments_stub() -> Dict[str, Any]:
    """Mission B funnel segmentation — in progress."""
    return {
        "status": "Mission B in progress",
        "segments": [],
        "note": (
            "Funnel segmentation results will be served here once "
            "Mission B is complete. See dashboard Tab 2 for preview."
        ),
    }
