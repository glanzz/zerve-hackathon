"""friction.py — per-user friction feature group.

compute_friction(events_df, cutoff_map) -> pd.DataFrame
    Columns: person_id, friction_events, friction_ratio

NOTE: total_events is computed inline; this module does NOT depend
on engagement.py output.
"""
from __future__ import annotations

import pandas as pd

# Must match CFG.FRICTION_EVENTS
FRICTION_EVENTS = ["$exception"]


def compute_friction(events_df: pd.DataFrame, cutoff_map: pd.Series) -> pd.DataFrame:
    """Compute friction features, respecting per-user cutoff timestamps.

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
        f"compute_friction: filtered {_n_drop:,} post-cutoff rows, "
        f"kept {_n_kept:,} for aggregation"
    )

    # ── Inline total_events per user ────────────────────────────────────────
    _totals = (
        _df.groupby("person_id")["event"]
        .count()
        .rename("_total_events")
        .reset_index()
    )

    # ── friction_events: count of FRICTION_EVENTS ────────────────────────────
    _fric = (
        _df[_df["event"].isin(FRICTION_EVENTS)]
        .groupby("person_id")["event"]
        .count()
        .rename("friction_events")
        .reset_index()
    )

    # ── Join ────────────────────────────────────────────────────────────────
    _result = _totals.merge(_fric, on="person_id", how="left")
    _result["friction_events"] = _result["friction_events"].fillna(0).astype(int)

    # friction_ratio = friction_events / total_events (0 if total==0)
    _result["friction_ratio"] = (
        _result["friction_events"]
        / _result["_total_events"].replace(0, float("nan"))
    ).fillna(0.0)

    # ── Spine alignment ─────────────────────────────────────────────────────
    _spine = cutoff_map.reset_index()[["person_id"]]
    _result = _spine.merge(_result[["person_id", "friction_events", "friction_ratio"]],
                           on="person_id", how="left")
    _result["friction_events"] = _result["friction_events"].fillna(0).astype(int)
    _result["friction_ratio"]  = _result["friction_ratio"].fillna(0.0)

    return _result[["person_id", "friction_events", "friction_ratio"]].reset_index(drop=True)
