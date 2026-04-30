"""mission_a_train.py — train LightGBM upgrade-predictor (Mission A).

Usage:
    python -m src.models.mission_a_train          # train + save artifacts
    from src.models.mission_a_train import train  # programmatic call
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

import lightgbm as lgb

from config.config import CFG
from src.models.evaluation import full_eval_report, plot_roc_curve, plot_pr_curve, plot_calibration_curve, plot_confusion_matrix, threshold_analysis

print(f"lightgbm version: {lgb.__version__}")

# ── constants ─────────────────────────────────────────────────────────────────
_SKIP_COLS = {"person_id", "user_cutoff_ts", "will_upgrade_in_7d", "split"}

_LGB_PARAMS = dict(
    n_estimators      = 1000,
    num_leaves        = 63,
    learning_rate     = 0.05,
    min_child_samples = 20,
    min_data_in_leaf  = 20,
    feature_fraction  = 0.8,
    bagging_fraction  = 0.8,
    bagging_freq      = 5,
    lambda_l1         = 0.1,
    lambda_l2         = 0.1,
    is_unbalance      = True,
    metric            = 'auc',
    random_state      = 42,
    n_jobs            = -1,
    verbose           = -1,
)


# ── helpers ───────────────────────────────────────────────────────────────────
def _make_pipeline(n_estimators: int = 1000) -> Pipeline:
    _params = dict(_LGB_PARAMS)
    _params["n_estimators"] = n_estimators
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model",   lgb.LGBMClassifier(**_params)),
    ])


def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.select_dtypes(include="number").columns
            if c not in _SKIP_COLS]


# ── main train function ───────────────────────────────────────────────────────
def train(
    use_tuned_params: bool = False,
    save_artifacts: bool = True,
) -> Tuple[Pipeline, Dict[str, Any]]:
    _t_start = time.time()

    # 1. Load data
    print("\n── Loading data ──")
    _fs = pd.read_parquet(CFG.FEATURE_STORE_PATH)
    _splits = pd.read_parquet(CFG.SPLITS_PATH)
    _splits["user_cutoff_ts"] = pd.to_datetime(
        _splits["user_cutoff_ts"], utc=True, errors="coerce"
    )
    _df = _fs.merge(_splits[["person_id", "split"]], on="person_id", how="inner")
    print(f"  Merged shape: {_df.shape}")

    # 2. Feature columns
    _feat_cols = _get_feature_cols(_df)
    print(f"\n── Feature columns ({len(_feat_cols)}) ──")
    for _c in _feat_cols:
        print(f"  {_c}")

    # 3. Build split masks
    _train_mask = _df["split"] == "train"
    _val_mask   = _df["split"] == "validation"
    _test_mask  = _df["split"] == "test"

    _X_trainval = _df.loc[_train_mask | _val_mask, _feat_cols].reset_index(drop=True)
    _y_trainval = _df.loc[_train_mask | _val_mask, "will_upgrade_in_7d"].astype(int).reset_index(drop=True)
    _pid_trainval = _df.loc[_train_mask | _val_mask, "person_id"].reset_index(drop=True)

    _X_test = _df.loc[_test_mask, _feat_cols].reset_index(drop=True)
    _y_test = _df.loc[_test_mask, "will_upgrade_in_7d"].astype(int).reset_index(drop=True)

    _n_train = _train_mask.sum()
    _n_val   = _val_mask.sum()
    _n_test  = _test_mask.sum()
    print(f"\n  train+val : {len(_y_trainval):,}  "
          f"({_y_trainval.sum()} positives, {_y_trainval.mean():.2%})")
    print(f"  test      : {_n_test:,}  "
          f"({_y_test.sum()} positives, {_y_test.mean():.2%})")

    # 4. Stratified 5-fold CV on train+val
    print("\n── Cross-validation (5-fold stratified) ──")
    _kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    _oof_proba = np.zeros(len(_y_trainval))
    _oof_fold  = np.zeros(len(_y_trainval), dtype=int)
    _fold_scores = []
    _best_iters  = []

    from sklearn.metrics import roc_auc_score, average_precision_score

    for _fold, (_tr_idx, _vl_idx) in enumerate(
            _kf.split(_X_trainval, _y_trainval), start=1):
        _Xtr, _ytr = _X_trainval.iloc[_tr_idx], _y_trainval.iloc[_tr_idx]
        _Xvl, _yvl = _X_trainval.iloc[_vl_idx], _y_trainval.iloc[_vl_idx]

        _pipe = _make_pipeline(n_estimators=1000)
        # fit imputer first, then call lgb with eval_set on transformed data
        _imp = SimpleImputer(strategy="median")
        _Xtr_t = _imp.fit_transform(_Xtr)
        _Xvl_t = _imp.transform(_Xvl)

        _clf = lgb.LGBMClassifier(**_LGB_PARAMS)
        print(f"  fold params (sample): metric={_LGB_PARAMS.get('metric')}, "
              f"is_unbalance={_LGB_PARAMS.get('is_unbalance')}, "
              f"min_data_in_leaf={_LGB_PARAMS.get('min_data_in_leaf')}")
        _clf.fit(
            _Xtr_t, _ytr,
            eval_set=[(_Xvl_t, _yvl)],
            eval_metric='auc',
            callbacks=[
                lgb.early_stopping(stopping_rounds=100, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        print(f"  fold {_fold}: best_iter={_clf.best_iteration_}, "
              f"best_score_={_clf.best_score_}")
        _best_iter = _clf.best_iteration_ if _clf.best_iteration_ > 0 else 1000
        _best_iters.append(_best_iter)

        # store fitted imputer+clf as mini-pipeline for this fold
        _fold_pipe = Pipeline([
            ("imputer", _imp),
            ("model",   _clf),
        ])
        _oof_proba[_vl_idx] = _fold_pipe.predict_proba(_Xvl)[:, 1]
        _oof_fold[_vl_idx]  = _fold

        _fold_auc = roc_auc_score(_yvl, _oof_proba[_vl_idx])
        _fold_pr  = average_precision_score(_yvl, _oof_proba[_vl_idx])
        _fold_scores.append({
            "fold": _fold,
            "n_val": len(_yvl),
            "n_pos": int(_yvl.sum()),
            "auc":   round(_fold_auc, 4),
            "pr_auc": round(_fold_pr, 4),
            "best_iteration": _best_iter,
        })
        print(f"  Fold {_fold}: AUC={_fold_auc:.4f}  PR-AUC={_fold_pr:.4f}"
              f"  best_iter={_best_iter}")

    _oof_auc = roc_auc_score(_y_trainval, _oof_proba)
    _oof_pr  = average_precision_score(_y_trainval, _oof_proba)
    print(f"\n{'='*50}")
    print(f"  OOF AUC    : {_oof_auc:.4f}")
    print(f"  OOF PR-AUC : {_oof_pr:.4f}")
    print(f"{'='*50}")

    # 5. Refit on all train+val using median best_iteration
    _med_iter = max(1, int(np.median(_best_iters)))
    print(f"\n── Refitting on full train+val (n_estimators={_med_iter}) ──")
    _final_pipe = _make_pipeline(n_estimators=_med_iter)
    _final_pipe.fit(_X_trainval, _y_trainval)

    # 6. Test evaluation
    print("\n── Test set evaluation ──")
    _eval_dir = Path(CFG.MODEL_DIR) / "evaluation"
    _eval_dir.mkdir(parents=True, exist_ok=True)

    _test_proba = _final_pipe.predict_proba(_X_test)[:, 1]
    _test_pred  = (_test_proba >= 0.5).astype(int)
    _test_report = full_eval_report(_y_test, _test_proba,
                                    threshold=0.5, dataset_name="test")

    plot_roc_curve(_y_test, _test_proba,
                   str(_eval_dir / "roc_curve.png"))
    plot_pr_curve(_y_test, _test_proba,
                  str(_eval_dir / "pr_curve.png"))
    plot_calibration_curve(_y_test, _test_proba,
                           str(_eval_dir / "calibration.png"))
    plot_confusion_matrix(_y_test, _test_pred,
                          str(_eval_dir / "confusion_matrix.png"))

    # 7. Threshold at max F1
    _ta = threshold_analysis(_y_test, _test_proba)
    _best_thresh = float(_ta.loc[_ta["f1"].idxmax(), "threshold"])
    print(f"\n  Threshold at max F1: {_best_thresh:.2f}")

    # 8. SHAP analysis
    print("\n── SHAP beeswarm (1000-sample) ──")
    try:
        import shap
        _sample_idx  = np.random.default_rng(42).choice(
            len(_X_test), size=min(1000, len(_X_test)), replace=False
        )
        _X_sample    = _X_test.iloc[_sample_idx]
        _X_sample_t  = _final_pipe[:-1].transform(_X_sample)
        _explainer   = shap.TreeExplainer(_final_pipe.named_steps["model"])
        _shap_vals   = _explainer.shap_values(_X_sample_t)
        if isinstance(_shap_vals, list):
            _shap_vals = _shap_vals[1]

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        _shap_fig, _shap_ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            _shap_vals, _X_sample_t,
            feature_names=_feat_cols,
            max_display=20,
            show=False,
        )
        _shap_path = Path(CFG.MODEL_DIR) / "shap_summary.png"
        plt.savefig(str(_shap_path), dpi=200, bbox_inches="tight")
        plt.close("all")
        print(f"  SHAP saved to {_shap_path}")
    except ImportError:
        print("  [WARNING] shap not installed — skipping beeswarm plot")
        _shap_path = None

    # 9. Save artifacts
    _elapsed = time.time() - _t_start
    _metadata = {
        "oof_auc":             round(_oof_auc, 4),
        "oof_pr_auc":          round(_oof_pr, 4),
        "test_auc":            _test_report["roc_auc"],
        "test_pr_auc":         _test_report["pr_auc"],
        "n_train":             int(_n_train),
        "n_val":               int(_n_val),
        "n_test":              int(_n_test),
        "n_features":          len(_feat_cols),
        "fold_scores":         _fold_scores,
        "best_iters":          _best_iters,
        "median_best_iter":    _med_iter,
        "threshold_at_max_f1": _best_thresh,
        "training_seconds":    round(_elapsed, 1),
        "trained_at":          datetime.now(timezone.utc).isoformat(),
    }

    if save_artifacts:
        _mdir = Path(CFG.MODEL_DIR)
        _mdir.mkdir(parents=True, exist_ok=True)

        _pkl_path = _mdir / "upgrade_predictor.pkl"
        with open(_pkl_path, "wb") as _f:
            pickle.dump(_final_pipe, _f)
        print(f"\n  Saved: {_pkl_path}  ({os.path.getsize(_pkl_path)/1e6:.2f} MB)")

        _fn_path = _mdir / "feature_names.pkl"
        with open(_fn_path, "wb") as _f:
            pickle.dump(_feat_cols, _f)
        print(f"  Saved: {_fn_path}")

        _meta_path = _mdir / "model_metadata.json"
        _meta_path.write_text(json.dumps(_metadata, indent=2), encoding="utf-8")
        print(f"  Saved: {_meta_path}")

        _oof_df = pd.DataFrame({
            "person_id":    _pid_trainval,
            "fold":         _oof_fold,
            "y_true":       _y_trainval,
            "y_pred_proba": _oof_proba,
        })
        _oof_df.to_csv(_mdir / "oof_predictions.csv", index=False)
        print(f"  Saved: {_mdir / 'oof_predictions.csv'}")

        _fold_df = pd.DataFrame(_fold_scores)
        _fold_df.to_csv(_mdir / "fold_scores.csv", index=False)
        print(f"  Saved: {_mdir / 'fold_scores.csv'}")

    # 10. Final summary
    print(f"\n{'='*50}")
    print(f"  TRAINING SUMMARY")
    print(f"{'='*50}")
    print(f"  Features       : {len(_feat_cols)}")
    print(f"  Train+val      : {len(_y_trainval):,}  ({_y_trainval.sum()} pos)")
    print(f"  Test           : {_n_test:,}  ({_y_test.sum()} pos)")
    print(f"  OOF  AUC       : {_oof_auc:.4f}")
    print(f"  OOF  PR-AUC    : {_oof_pr:.4f}")
    print(f"  Test AUC       : {_test_report['roc_auc']:.4f}")
    print(f"  Test PR-AUC    : {_test_report['pr_auc']:.4f}")
    print(f"  Best thresh    : {_best_thresh:.2f}")
    print(f"  Elapsed        : {_elapsed:.1f}s")
    print(f"{'='*50}")

    return _final_pipe, _metadata


# ── __main__ ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time as _time
    _t0 = _time.time()
    _pipeline, _meta = train(use_tuned_params=False, save_artifacts=True)
    print(f"\nTotal wall time: {_time.time() - _t0:.1f}s")
