"""mission_a_predict.py — inference wrapper for Mission A upgrade predictor.

Functions:
    load_model()      -> (Pipeline, feature_names, metadata)
    predict_single()  -> dict
    predict_batch()   -> pd.DataFrame
    get_risk_tier()   -> (tier_label, recommended_action)
"""
from __future__ import annotations

import json
import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from config.config import CFG

# ── Risk tier thresholds ──────────────────────────────────────────────────────
_TIERS = [
    (0.75, "HOT_LEAD",  "Trigger in-app upgrade modal"),
    (0.50, "WARM_LEAD", "Send personalized feature highlight email"),
    (0.25, "NURTURE",   "Enroll in drip campaign"),
    (0.00, "COLD",      "Focus on Aha moment activation"),
]


# ── Module-level cache (load once) ────────────────────────────────────────────
_MODEL_CACHE: Dict[str, Any] = {}


def load_model() -> Tuple[Any, List[str], Dict[str, Any]]:
    """Load pipeline, feature names, and metadata. Cached after first call."""
    if _MODEL_CACHE:
        return (
            _MODEL_CACHE["pipeline"],
            _MODEL_CACHE["feature_names"],
            _MODEL_CACHE["metadata"],
        )

    # Resolve artifact directory: prefer sibling 'models/' next to this file
    # (correct in deployed container), fall back to CFG.MODEL_DIR for dev.
    _script_dir = Path(__file__).resolve().parent
    _mdir_candidate = _script_dir / "models"
    _mdir = _mdir_candidate if _mdir_candidate.exists() else Path(CFG.MODEL_DIR)

    _pkl_path = _mdir / "upgrade_predictor.pkl"
    if not _pkl_path.exists():
        raise FileNotFoundError(
            f"Model not found at {_pkl_path}. "
            "Run src/models/mission_a_train.py first."
        )
    with open(_pkl_path, "rb") as _f:
        _pipeline = pickle.load(_f)

    _fn_path = _mdir / "feature_names.pkl"
    with open(_fn_path, "rb") as _f:
        _feature_names = pickle.load(_f)

    _meta_path = _mdir / "model_metadata.json"
    _metadata = json.loads(_meta_path.read_text(encoding="utf-8"))

    _MODEL_CACHE["pipeline"]      = _pipeline
    _MODEL_CACHE["feature_names"] = _feature_names
    _MODEL_CACHE["metadata"]      = _metadata

    print(f"Model loaded: {len(_feature_names)} features, "
          f"test_auc={_metadata.get('test_auc', 'n/a')}, "
          f"trained_at={_metadata.get('trained_at', 'n/a')}")
    return _pipeline, _feature_names, _metadata


def get_risk_tier(prob: float) -> Tuple[str, str]:
    """Map a probability to (risk_tier_label, recommended_action)."""
    for _threshold, _label, _action in _TIERS:
        if prob >= _threshold:
            return _label, _action
    return "COLD", "Focus on Aha moment activation"


def predict_single(user_features: Dict[str, float]) -> Dict[str, Any]:
    """Predict upgrade probability for a single user dict.

    Returns:
        upgrade_probability, upgrade_risk_tier, recommended_action,
        model_version, feature_count, top_3_features (with values).
    """
    _pipeline, _feat_cols, _meta = load_model()

    # Build single-row DataFrame aligned to training feature order
    _row = {_c: user_features.get(_c, np.nan) for _c in _feat_cols}
    _X   = pd.DataFrame([_row])[_feat_cols]

    _prob = float(_pipeline.predict_proba(_X)[0, 1])
    _tier, _action = get_risk_tier(_prob)

    # Top 3 features by absolute value (after imputation)
    _X_t = _pipeline[:-1].transform(_X)
    _importances = _pipeline.named_steps["model"].feature_importances_
    _top3_idx = np.argsort(_importances)[::-1][:3]
    _top3 = [
        {"feature": _feat_cols[_i], "value": float(_X_t[0, _i]),
         "importance": float(_importances[_i])}
        for _i in _top3_idx
    ]

    return {
        "upgrade_probability": round(_prob, 4),
        "upgrade_risk_tier":   _tier,
        "recommended_action":  _action,
        "model_version":       _meta.get("trained_at", "unknown"),
        "feature_count":       len(_feat_cols),
        "top_3_features":      _top3,
    }


def predict_batch(feature_store: pd.DataFrame) -> pd.DataFrame:
    """Vectorized prediction on a feature store DataFrame.

    Adds columns: upgrade_probability, upgrade_risk_tier, recommended_action.
    Returns a copy with those columns appended.
    """
    _pipeline, _feat_cols, _ = load_model()

    # Align columns — fill missing with NaN
    _missing = [_c for _c in _feat_cols if _c not in feature_store.columns]
    if _missing:
        print(f"  [WARNING] {len(_missing)} feature cols missing from input; "
              f"filling with NaN: {_missing[:5]}{'...' if len(_missing)>5 else ''}")

    _X = feature_store.reindex(columns=_feat_cols)
    _proba = _pipeline.predict_proba(_X)[:, 1]

    _result = feature_store.copy()
    _result["upgrade_probability"] = _proba.round(4)
    _tiers = [get_risk_tier(_p) for _p in _proba]
    _result["upgrade_risk_tier"]   = [_t[0] for _t in _tiers]
    _result["recommended_action"]  = [_t[1] for _t in _tiers]
    return _result


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Loading model ──")
    _pipe, _feat_names, _meta = load_model()
    print(f"  Features     : {len(_feat_names)}")
    print(f"  OOF AUC      : {_meta.get('oof_auc', 'n/a')}")
    print(f"  Test AUC     : {_meta.get('test_auc', 'n/a')}")

    print("\n── Batch prediction on first 5 rows of feature store ──")
    _fs = pd.read_parquet(CFG.FEATURE_STORE_PATH)
    _preds = predict_batch(_fs.head(5))
    _show_cols = ["person_id", "upgrade_probability",
                  "upgrade_risk_tier", "recommended_action"]
    print(_preds[_show_cols].to_string(index=False))

    print("\n── Single prediction ──")
    _sample_user = {_c: float(_fs[_c].median()) for _c in _feat_names
                    if _c in _fs.columns}
    _single = predict_single(_sample_user)
    for _k, _v in _single.items():
        print(f"  {_k}: {_v}")

    print("\npredict.py smoke test PASSED")
