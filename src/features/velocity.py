"""velocity.py — per-user velocity / momentum feature group.

compute_velocity(events_df, cutoff_map) -> pd.DataFrame
    Columns: person_id, events_last_7d, events_last_14d, events_last_30d,
             events_last_14_to_7d, wow_growth, is_accelerating
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def compute_velocity(events_df: pd.DataFrame, cutoff_map: pd.Series) -> pd.DataFrame:
    """Compute velocity features, respecting per-user cutoff timestamps.

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
        f"compute_velocity: filtered {_n_drop:,} post-cutoff rows, "
        f"kept {_n_kept:,} for aggregation"
    )

    # ── Ensure both columns are tz-aware UTC datetimes before subtraction ───
    # cutoff_ts comes from labels parquet (may be object/string after merge)
    _df["timestamp"]  = pd.to_datetime(_df["timestamp"],  utc=True, errors="coerce")
    _df["cutoff_ts"]  = pd.to_datetime(_df["cutoff_ts"],  utc=True, errors="coerce")

    # ── Time deltas (seconds for precision, convert to days) ────────────────
    _df["_delta_days"] = (
        (_df["cutoff_ts"] - _df["timestamp"]).dt.total_seconds() / 86400.0
    )

    # ── Window flags ────────────────────────────────────────────────────────
    _df["_in_7d"]  = (_df["_delta_days"] >= 0) & (_df["_delta_days"] <  7)
    _df["_in_14d"] = (_df["_delta_days"] >= 0) & (_df["_delta_days"] < 14)
    _df["_in_30d"] = (_df["_delta_days"] >= 0) & (_df["_delta_days"] < 30)

    # ── Aggregate per user ──────────────────────────────────────────────────
    _agg = (
        _df.groupby("person_id")
        .agg(
            events_last_7d=("_in_7d",  "sum"),
            events_last_14d=("_in_14d", "sum"),
            events_last_30d=("_in_30d", "sum"),
        )
        .reset_index()
    )

    # ── Derived velocity features ───────────────────────────────────────────
    _agg["events_last_14_to_7d"] = _agg["events_last_14d"] - _agg["events_last_7d"]
    _agg["wow_growth"] = (
        (_agg["events_last_7d"] - _agg["events_last_14_to_7d"])
        / (_agg["events_last_14_to_7d"] + 1.0)
    )
    _agg["is_accelerating"] = (_agg["wow_growth"] > 0.20).astype(int)

    # ── Spine alignment ─────────────────────────────────────────────────────
    _spine = cutoff_map.reset_index()[["person_id"]]
    _result = _spine.merge(_agg, on="person_id", how="left")
    _fill_cols = ["events_last_7d", "events_last_14d", "events_last_30d",
                  "events_last_14_to_7d", "wow_growth", "is_accelerating"]
    _result[_fill_cols] = _result[_fill_cols].fillna(0)

    return _result[["person_id"] + _fill_cols].reset_index(drop=True)
