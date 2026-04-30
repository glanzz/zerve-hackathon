"""depth.py — per-user feature depth / AHA-moment feature group.

compute_depth(events_df, cutoff_map) -> pd.DataFrame
    Columns: person_id, feature_breadth, hit_aha_moment,
             ai_generation_count, run_block_count, agent_chat_count
"""
from __future__ import annotations

import pandas as pd

# Must match CFG.CORE_EVENTS_FOR_DEPTH
CORE_EVENTS = [
    "agent_new_chat",
    "$ai_generation",
    "run_block",
    "agent_tool_call_create_block_tool",
    "agent_tool_call_run_block_tool",
    "agent_tool_call_get_block_tool",
]


def compute_depth(events_df: pd.DataFrame, cutoff_map: pd.Series) -> pd.DataFrame:
    """Compute depth features, respecting per-user cutoff timestamps.

    Parameters
    ----------
    events_df : raw events DataFrame with columns [person_id, event, timestamp]
    cutoff_map : pd.Series indexed by person_id, values = cutoff_ts

    Returns
    -------
    pd.DataFrame with one row per person_id in cutoff_map
    """
    # ── Merge & cutoff filter ───────────────────────────────────────────────
    _df = events_df.merge(
        cutoff_map.rename("cutoff_ts").reset_index(),
        on="person_id",
        how="inner",
    )
    _df = _df[_df["timestamp"] <= _df["cutoff_ts"]].copy()

    _n_kept = len(_df)
    _n_drop = len(events_df) - _n_kept
    print(
        f"compute_depth: filtered {_n_drop:,} post-cutoff rows, "
        f"kept {_n_kept:,} for aggregation"
    )

    # ── feature_breadth: distinct CORE_EVENTS fired per user ───────────────
    _core_df = _df[_df["event"].isin(CORE_EVENTS)]
    _breadth = (
        _core_df.groupby("person_id")["event"]
        .nunique()
        .rename("feature_breadth")
        .reset_index()
    )

    # ── per-event counts ────────────────────────────────────────────────────
    _pivot = (
        _df[_df["event"].isin(["$ai_generation", "run_block", "agent_new_chat"])]
        .groupby(["person_id", "event"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    # Ensure all three columns exist even if event never appeared
    for _col in ["$ai_generation", "run_block", "agent_new_chat"]:
        if _col not in _pivot.columns:
            _pivot[_col] = 0

    _pivot = _pivot.rename(columns={
        "$ai_generation": "ai_generation_count",
        "run_block":       "run_block_count",
        "agent_new_chat":  "agent_chat_count",
    })

    # ── Join & derive hit_aha_moment ────────────────────────────────────────
    _result = _breadth.merge(_pivot[["person_id", "ai_generation_count",
                                     "run_block_count", "agent_chat_count"]],
                             on="person_id", how="outer")

    # ── Spine alignment ─────────────────────────────────────────────────────
    _spine = cutoff_map.reset_index()[["person_id"]]
    _result = _spine.merge(_result, on="person_id", how="left")
    _fill_cols = ["feature_breadth", "ai_generation_count",
                  "run_block_count", "agent_chat_count"]
    _result[_fill_cols] = _result[_fill_cols].fillna(0).astype(int)
    _result["hit_aha_moment"] = (_result["feature_breadth"] >= 2).astype(int)

    _out_cols = ["person_id", "feature_breadth", "hit_aha_moment",
                 "ai_generation_count", "run_block_count", "agent_chat_count"]
    return _result[_out_cols].reset_index(drop=True)
