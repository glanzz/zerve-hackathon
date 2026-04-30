"""
funnel_report.py — Combined Mission A + Mission B report generator.

Combines upgrade predictions (Mission A) with funnel segments (Mission B)
to produce a per-segment business intelligence report saved as JSON + PNG.

Output:
  models/funnel_report.json  — full report dict
  models/segment_upgrade_proba.png — bar chart of mean upgrade prob per segment
"""
from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


_BG  = "#1D1D20"
_FG  = "#fbfbff"
_SUB = "#909094"
_PALETTE = [
    "#A1C9F4", "#FFB482", "#8DE5A1", "#FF9F9B",
    "#D0BBFF", "#1F77B4", "#9467BD", "#8C564B",
    "#C49C94", "#E377C2",
]


def generate_full_report(
    feature_store: pd.DataFrame,
    df_with_segments: pd.DataFrame,
    df_with_predictions: Optional[pd.DataFrame] = None,
    save_dir: str = "models",
) -> Dict[str, Any]:
    """
    Combine Mission A predictions with Mission B segments per-segment report.

    For each segment computes:
      - user_count + pct_of_total
      - mean upgrade_probability (from Mission A, if available)
      - top 3 most diagnostic features (highest mean value within segment)
      - recommended business action
      - conversion_rate: % of segment users who actually upgraded

    Saves:
      - models/funnel_report.json
      - models/segment_upgrade_proba.png

    Args:
        feature_store: Base feature store DataFrame.
        df_with_segments: Output of assign_segments() — has 'funnel_segment'.
        df_with_predictions: Optional DataFrame with 'person_id' and
                             'upgrade_probability' columns (Mission A output).
        save_dir: Directory to save output files.

    Returns:
        Report dictionary keyed by segment name.
    """
    import sys, os as _os
    sys.path.insert(0, _os.getcwd())
    from config.config import CFG
    from src.funnel.segments import SEGMENT_DEFINITIONS, SEGMENT_ORDER

    if "funnel_segment" not in df_with_segments.columns:
        raise KeyError("generate_full_report: 'funnel_segment' column missing — run assign_segments first.")

    os.makedirs(save_dir, exist_ok=True)

    # ── Merge predictions if provided ─────────────────────────────────────
    df = df_with_segments.copy()
    if df_with_predictions is not None and "upgrade_probability" in df_with_predictions.columns:
        preds = df_with_predictions[["person_id", "upgrade_probability"]].drop_duplicates("person_id")
        df = df.merge(preds, on="person_id", how="left")
        has_predictions = True
    else:
        has_predictions = False
        df["upgrade_probability"] = np.nan

    # If will_upgrade_in_7d is present, use it for actual conversion rate
    has_actual = "will_upgrade_in_7d" in df.columns

    # ── Feature columns for "top diagnostics" ─────────────────────────────
    exclude_cols = {
        "person_id", "will_upgrade_in_7d", "user_cutoff_ts",
        "funnel_segment", "cluster_id", "upgrade_probability",
    }
    feature_cols = [c for c in df.columns
                    if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    n_total = len(df)
    report: Dict[str, Any] = {}

    all_segs = [s for s in SEGMENT_ORDER if s in df["funnel_segment"].unique()]
    if "Unclassified" in df["funnel_segment"].unique():
        all_segs.append("Unclassified")

    for seg in all_segs:
        mask = df["funnel_segment"] == seg
        seg_df = df[mask]
        n_seg  = int(mask.sum())

        if n_seg == 0:
            continue

        pct = round(n_seg / n_total * 100, 2)

        # Mean upgrade probability
        mean_prob = (
            float(seg_df["upgrade_probability"].mean())
            if has_predictions and seg_df["upgrade_probability"].notna().any()
            else None
        )

        # Actual conversion rate
        conv_rate = (
            float(seg_df["will_upgrade_in_7d"].mean())
            if has_actual
            else None
        )

        # Top 3 most diagnostic features (highest mean within segment vs overall mean)
        seg_means    = seg_df[feature_cols].mean()
        overall_means = df[feature_cols].mean()
        # Lift = (seg_mean - overall_mean) / (overall_mean + 1e-9)
        lift = (seg_means - overall_means) / (overall_means.abs() + 1e-9)
        top3 = lift.abs().nlargest(3).index.tolist()
        top3_vals = {f: round(float(seg_means[f]), 3) for f in top3}

        # Business action from SEGMENT_DEFINITIONS (or generic for Unclassified)
        seg_def = SEGMENT_DEFINITIONS.get(seg, {})
        action  = seg_def.get("action", "Review and classify manually")
        upgrade_likelihood = seg_def.get("upgrade_likelihood", "Unknown")

        report[seg] = {
            "user_count"         : n_seg,
            "pct_of_total"       : pct,
            "mean_upgrade_prob"  : mean_prob,
            "actual_conversion_rate": conv_rate,
            "upgrade_likelihood" : upgrade_likelihood,
            "top_diagnostic_features": top3_vals,
            "recommended_action" : action,
            "description"        : seg_def.get("description", ""),
        }

    # ── Print summary table ────────────────────────────────────────────────
    print("\n" + "="*70)
    print("  FUNNEL REPORT SUMMARY")
    print("="*70)
    print(f"  {'Segment':<22} {'N':>6} {'%':>6} {'Conv%':>7} {'AvgProb':>8}  Action")
    print(f"  {'-'*70}")
    for seg, r in report.items():
        conv_s = f"{r['actual_conversion_rate']*100:.1f}%" if r['actual_conversion_rate'] is not None else "  N/A"
        prob_s = f"{r['mean_upgrade_prob']:.3f}" if r['mean_upgrade_prob'] is not None else "   N/A"
        action_short = r['recommended_action'][:35]
        print(f"  {seg:<22} {r['user_count']:>6,} {r['pct_of_total']:>5.1f}% {conv_s:>7} {prob_s:>8}  {action_short}")
    print("="*70)

    # ── Save JSON ──────────────────────────────────────────────────────────
    json_path = Path(save_dir) / "funnel_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n  Saved funnel_report.json → {json_path}")

    # ── Bar chart: mean upgrade probability per segment ────────────────────
    segs_with_prob = [(s, r) for s, r in report.items()
                      if r["mean_upgrade_prob"] is not None]

    if segs_with_prob or has_actual:
        seg_names = [r[0] for r in segs_with_prob] if segs_with_prob else list(report.keys())
        if has_actual and not segs_with_prob:
            y_vals  = [report[s]["actual_conversion_rate"] * 100 for s in seg_names]
            y_label = "Actual Conversion Rate (%)"
            title   = "Actual Conversion Rate by Funnel Segment"
        else:
            y_vals  = [r[1]["mean_upgrade_prob"] * 100 for r in segs_with_prob]
            seg_names = [r[0] for r in segs_with_prob]
            y_label = "Mean Predicted Upgrade Probability (%)"
            title   = "Mean Upgrade Probability by Funnel Segment (Mission A × B)"

        colours = [_PALETTE[i % len(_PALETTE)] for i in range(len(seg_names))]

        fig, ax = plt.subplots(figsize=(max(8, len(seg_names) * 1.4), 5),
                               facecolor=_BG)
        ax.set_facecolor(_BG)

        bars = ax.bar(seg_names, y_vals, color=colours, width=0.6, zorder=3)

        # Value labels on bars
        for bar, v in zip(bars, y_vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(y_vals) * 0.01,
                f"{v:.1f}%",
                ha="center", va="bottom", color=_FG, fontsize=9, fontweight="bold",
            )

        ax.set_title(title, color=_FG, fontsize=12, pad=10)
        ax.set_xlabel("Funnel Segment", color=_SUB, labelpad=8)
        ax.set_ylabel(y_label, color=_SUB, labelpad=8)
        ax.tick_params(axis="x", colors=_FG, rotation=20, labelsize=9)
        ax.tick_params(axis="y", colors=_FG, labelsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.3, color=_SUB)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_edgecolor(_SUB)

        fig.tight_layout()
        bar_path = Path(save_dir) / "segment_upgrade_proba.png"
        fig.savefig(bar_path, dpi=120, bbox_inches="tight", facecolor=_BG)
        plt.close(fig)
        print(f"  Saved segment_upgrade_proba.png → {bar_path}")

    return report


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.getcwd())
    import pandas as pd
    from config.config import CFG
    from src.funnel.segments import assign_segments

    print("Loading feature store...")
    fs = pd.read_parquet(CFG.FEATURE_STORE_PATH)
    seg_df = assign_segments(fs)

    report = generate_full_report(
        feature_store=fs,
        df_with_segments=seg_df,
        df_with_predictions=None,
    )
    print(f"\nReport keys: {list(report.keys())}")
