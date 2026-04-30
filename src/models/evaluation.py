"""evaluation.py — model evaluation utilities for Mission A.

Functions:
    full_eval_report(y_true, y_pred_proba, threshold, dataset_name) -> dict
    plot_roc_curve(y_true, y_pred_proba, save_path)
    plot_pr_curve(y_true, y_pred_proba, save_path)
    plot_calibration_curve(y_true, y_pred_proba, save_path)
    plot_confusion_matrix(y_true, y_pred, save_path)
    threshold_analysis(y_true, y_pred_proba) -> pd.DataFrame
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    brier_score_loss,
    log_loss,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.calibration import calibration_curve

# ── Zerve design-system colours ──────────────────────────────────────────────
_BG      = "#1D1D20"
_FG      = "#fbfbff"
_BLUE    = "#A1C9F4"
_ORANGE  = "#FFB482"
_GREEN   = "#8DE5A1"
_CORAL   = "#FF9F9B"
_GOLD    = "#ffd400"


def _apply_dark_style(fig: plt.Figure, ax: plt.Axes) -> None:
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG)
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor("#909094")
    ax.legend(facecolor="#2a2a2e", edgecolor="#909094",
              labelcolor=_FG, fontsize=9) if ax.get_legend() else None


# ── full_eval_report ──────────────────────────────────────────────────────────
def full_eval_report(
    y_true,
    y_pred_proba,
    threshold: float = 0.5,
    dataset_name: str = "validation",
) -> Dict[str, Any]:
    _y_true = np.asarray(y_true)
    _y_prob = np.asarray(y_pred_proba)
    _y_pred = (_y_prob >= threshold).astype(int)

    _roc_auc  = roc_auc_score(_y_true, _y_prob)
    _pr_auc   = average_precision_score(_y_true, _y_prob)
    _f1       = f1_score(_y_true, _y_pred, zero_division=0)
    _prec     = precision_score(_y_true, _y_pred, zero_division=0)
    _rec      = recall_score(_y_true, _y_pred, zero_division=0)
    _brier    = brier_score_loss(_y_true, _y_prob)
    _logloss  = log_loss(_y_true, _y_prob)
    _n_pos    = int(_y_true.sum())
    _n_total  = len(_y_true)
    _pos_rate = _n_pos / _n_total

    _result = {
        "roc_auc":        round(_roc_auc, 4),
        "pr_auc":         round(_pr_auc, 4),
        "f1":             round(_f1, 4),
        "precision":      round(_prec, 4),
        "recall":         round(_rec, 4),
        "brier_score":    round(_brier, 4),
        "log_loss":       round(_logloss, 4),
        "n_positives":    _n_pos,
        "n_total":        _n_total,
        "positive_rate":  round(_pos_rate, 4),
        "threshold_used": threshold,
    }

    _pad = max(len(k) for k in _result)
    print(f"\n── {dataset_name} ──")
    for _k, _v in _result.items():
        print(f"  {_k:<{_pad}} : {_v}")

    return _result


# ── plot_roc_curve ────────────────────────────────────────────────────────────
def plot_roc_curve(y_true, y_pred_proba, save_path) -> None:
    _fpr, _tpr, _ = roc_curve(np.asarray(y_true), np.asarray(y_pred_proba))
    _auc = roc_auc_score(np.asarray(y_true), np.asarray(y_pred_proba))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(_fpr, _tpr, color=_BLUE, lw=2, label=f"ROC (AUC = {_auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#909094", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    _apply_dark_style(fig, ax)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close(fig)


# ── plot_pr_curve ─────────────────────────────────────────────────────────────
def plot_pr_curve(y_true, y_pred_proba, save_path) -> None:
    _prec, _rec, _ = precision_recall_curve(
        np.asarray(y_true), np.asarray(y_pred_proba)
    )
    _pr_auc = average_precision_score(
        np.asarray(y_true), np.asarray(y_pred_proba)
    )
    _baseline = np.asarray(y_true).mean()

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(_rec, _prec, color=_ORANGE, lw=2,
            label=f"PR (AP = {_pr_auc:.3f})")
    ax.axhline(_baseline, color="#909094", lw=1, linestyle="--",
               label=f"Baseline ({_baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision–Recall Curve")
    ax.legend()
    _apply_dark_style(fig, ax)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close(fig)


# ── plot_calibration_curve ────────────────────────────────────────────────────
def plot_calibration_curve(y_true, y_pred_proba, save_path) -> None:
    _frac_pos, _mean_pred = calibration_curve(
        np.asarray(y_true), np.asarray(y_pred_proba),
        n_bins=10, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    ax.plot(_mean_pred, _frac_pos, "s-", color=_GREEN, lw=2,
            label="Model calibration")
    ax.plot([0, 1], [0, 1], "--", color="#909094", lw=1, label="Perfect")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    _apply_dark_style(fig, ax)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close(fig)


# ── plot_confusion_matrix ─────────────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path) -> None:
    _cm = confusion_matrix(np.asarray(y_true), np.asarray(y_pred))

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    try:
        import seaborn as sns
        sns.heatmap(
            _cm, annot=True, fmt="d", cmap="Blues",
            ax=ax,
            annot_kws={"color": _FG, "size": 14},
            linewidths=0.5,
        )
    except ImportError:
        _im = ax.imshow(_cm, interpolation="nearest", cmap="Blues")
        for _i in range(_cm.shape[0]):
            for _j in range(_cm.shape[1]):
                ax.text(_j, _i, str(_cm[_i, _j]),
                        ha="center", va="center", color=_FG, fontsize=14)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    ax.set_xticklabels(["Neg", "Pos"])
    ax.set_yticklabels(["Neg", "Pos"], rotation=0)
    _apply_dark_style(fig, ax)
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=_BG, edgecolor="none")
    plt.close(fig)


# ── threshold_analysis ────────────────────────────────────────────────────────
def threshold_analysis(y_true, y_pred_proba) -> pd.DataFrame:
    _y_true = np.asarray(y_true)
    _y_prob = np.asarray(y_pred_proba)
    _n      = len(_y_true)

    _rows = []
    for _t in np.arange(0.10, 0.95, 0.05):
        _pred = (_y_prob >= _t).astype(int)
        _n_flagged = int(_pred.sum())
        _rows.append({
            "threshold":  round(float(_t), 2),
            "precision":  round(precision_score(_y_true, _pred, zero_division=0), 4),
            "recall":     round(recall_score(_y_true, _pred, zero_division=0), 4),
            "f1":         round(f1_score(_y_true, _pred, zero_division=0), 4),
            "n_flagged":  _n_flagged,
            "pct_flagged": round(_n_flagged / _n * 100, 2),
        })
    return pd.DataFrame(_rows)


# ── smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os
    _rng = np.random.default_rng(42)
    _n   = 1000
    _y_true = (_rng.random(_n) < 0.05).astype(int)
    _y_prob = np.clip(_y_true + _rng.normal(0, 0.3, _n), 0, 1)

    print("── full_eval_report ──")
    _report = full_eval_report(_y_true, _y_prob, threshold=0.5,
                               dataset_name="smoke_test")
    assert "roc_auc" in _report

    print("\n── threshold_analysis ──")
    _ta = threshold_analysis(_y_true, _y_prob)
    print(_ta.to_string(index=False))

    _tmp = tempfile.gettempdir()
    print("\n── plots ──")
    plot_roc_curve(_y_true, _y_prob,
                   os.path.join(_tmp, "roc.png"))
    print(f"  roc saved to {_tmp}/roc.png")
    plot_pr_curve(_y_true, _y_prob,
                  os.path.join(_tmp, "pr.png"))
    print(f"  pr  saved to {_tmp}/pr.png")
    plot_calibration_curve(_y_true, _y_prob,
                           os.path.join(_tmp, "cal.png"))
    print(f"  cal saved to {_tmp}/cal.png")
    plot_confusion_matrix(_y_true, (_y_prob >= 0.5).astype(int),
                          os.path.join(_tmp, "cm.png"))
    print(f"  cm  saved to {_tmp}/cm.png")

    print("\nevaluation.py smoke test PASSED")
