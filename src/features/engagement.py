"""engagement.py — per-user engagement feature group.

compute_engagement(events_df, cutoff_map) -> pd.DataFrame
    Columns: person_id, total_events, unique_event_types,
             total_sessions, avg_events_per_session, active_days
"""
from __future__ import annotations

import pandas as pd


def compute_engagement(events_df: pd.DataFrame, cutoff_map: pd.Series) -> pd.DataFrame:
    """Compute engagement features, respecting per-user cutoff timestamps.

    Parameters
    ----------
    events_df : raw events DataFrame.
                'properties.canvas_id' is expected as a top-level column
                (flat parquet schema — NOT a nested dict).
    cutoff_map : pd.Series indexed by person_id, values = cutoff_ts (datetime64[ns, UTC])

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
        f"compute_engagement: filtered {_n_drop:,} post-cutoff rows, "
        f"kept {_n_kept:,} for aggregation"
    )

    # ── total_events, unique_event_types, active_days ───────────────────────
    _base = (
        _df.groupby("person_id")
        .agg(
            total_events=("event", "count"),
            unique_event_types=("event", "nunique"),
            active_days=("timestamp", lambda s: s.dt.floor("D").nunique()),
        )
        .reset_index()
    )

    # ── total_sessions: 'properties.canvas_id' is a TOP-LEVEL flat column ──
    # (parquet schema exports nested props as dot-separated column names)
    if "properties.canvas_id" in _df.columns:
        _sessions = (
            _df[_df["properties.canvas_id"].notna()]
            .groupby("person_id")["properties.canvas_id"]
            .nunique()
            .rename("total_sessions")
            .reset_index()
        )
    elif "properties" in _df.columns:
        # fallback: legacy dict-in-column layout
        _canvas = _df["properties"].apply(
            lambda p: p.get("canvas_id") if isinstance(p, dict) else None
        )
        _sess_df = _df.copy()
        _sess_df["_canvas_id"] = _canvas
        _sessions = (
            _sess_df[_sess_df["_canvas_id"].notna()]
            .groupby("person_id")["_canvas_id"]
            .nunique()
            .rename("total_sessions")
            .reset_index()
        )
    else:
        print("  [WARNING] neither 'properties.canvas_id' nor 'properties' column found; total_sessions = 0")
        _sessions = _df[["person_id"]].drop_duplicates().copy()
        _sessions["total_sessions"] = 0

    # ── Join sessions back ──────────────────────────────────────────────────
    _result = _base.merge(_sessions, on="person_id", how="left")
    _result["total_sessions"] = _result["total_sessions"].fillna(0).astype(int)

    # ── avg_events_per_session ──────────────────────────────────────────────
    _result["avg_events_per_session"] = (
        _result["total_events"] / _result["total_sessions"].replace(0, float("nan"))
    ).fillna(0.0)

    # ── Spine alignment: ensure every cutoff user is present ───────────────
    _spine = cutoff_map.reset_index()[["person_id"]]
    _result = _spine.merge(_result, on="person_id", how="left")
    _fill_cols = ["total_events", "unique_event_types", "total_sessions",
                  "avg_events_per_session", "active_days"]
    _result[_fill_cols] = _result[_fill_cols].fillna(0)

    return _result[["person_id"] + _fill_cols].reset_index(drop=True)
