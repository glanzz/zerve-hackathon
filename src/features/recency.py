"""recency.py — per-user recency & tenure feature group.

compute_recency(events_df, cutoff_map) -> pd.DataFrame
    Columns: person_id, days_since_last_event, days_since_first_event,
             account_age_days
"""
from __future__ import annotations

import warnings

import pandas as pd


# Must match CFG.TENURE_ANCHOR_EVENT
TENURE_ANCHOR_EVENT = "new_user_created"


def compute_recency(events_df: pd.DataFrame, cutoff_map: pd.Series) -> pd.DataFrame:
    """Compute recency and tenure features, respecting per-user cutoffs.

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
        f"compute_recency: filtered {_n_drop:,} post-cutoff rows, "
        f"kept {_n_kept:,} for aggregation"
    )

    # ── Ensure both datetime columns are tz-aware UTC ───────────────────────
    _df["timestamp"] = pd.to_datetime(_df["timestamp"], utc=True, errors="coerce")
    _df["cutoff_ts"] = pd.to_datetime(_df["cutoff_ts"], utc=True, errors="coerce")

    # ── days_since_last_event & days_since_first_event ──────────────────────
    _base = (
        _df.groupby("person_id")
        .agg(
            _max_ts=("timestamp", "max"),
            _min_ts=("timestamp", "min"),
            _cutoff=("cutoff_ts", "first"),
        )
        .reset_index()
    )
    # Ensure aggregated columns are also tz-aware
    _base["_max_ts"] = pd.to_datetime(_base["_max_ts"], utc=True)
    _base["_min_ts"] = pd.to_datetime(_base["_min_ts"], utc=True)
    _base["_cutoff"] = pd.to_datetime(_base["_cutoff"], utc=True)

    _base["days_since_last_event"] = (
        (_base["_cutoff"] - _base["_max_ts"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0)
    _base["days_since_first_event"] = (
        (_base["_cutoff"] - _base["_min_ts"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0)

    # ── account_age_days: first TENURE_ANCHOR_EVENT, fallback min(ts) ───────
    _anchor = (
        _df[_df["event"] == TENURE_ANCHOR_EVENT]
        .groupby("person_id")["timestamp"]
        .min()
        .rename("_anchor_ts")
        .reset_index()
    )
    _base = _base.merge(_anchor, on="person_id", how="left")
    _base["_anchor_ts"] = pd.to_datetime(_base["_anchor_ts"], utc=True)

    _n_fallback = _base["_anchor_ts"].isna().sum()
    if _n_fallback > 0:
        warnings.warn(
            f"compute_recency: {_n_fallback} users have no '{TENURE_ANCHOR_EVENT}' event; "
            f"falling back to min(timestamp) for account_age_days.",
            stacklevel=2,
        )
        print(
            f"  [WARNING] {_n_fallback} users missing '{TENURE_ANCHOR_EVENT}'; "
            f"using min(timestamp) as tenure anchor."
        )

    _base["_anchor_ts"] = _base["_anchor_ts"].fillna(_base["_min_ts"])
    _base["account_age_days"] = (
        (_base["_cutoff"] - _base["_anchor_ts"]).dt.total_seconds() / 86400.0
    ).clip(lower=0.0)

    # ── Spine alignment ─────────────────────────────────────────────────────
    _spine = cutoff_map.reset_index()[["person_id"]]
    _result = _spine.merge(
        _base[["person_id", "days_since_last_event", "days_since_first_event",
               "account_age_days"]],
        on="person_id",
        how="left",
    )
    _fill_cols = ["days_since_last_event", "days_since_first_event", "account_age_days"]
    _result[_fill_cols] = _result[_fill_cols].fillna(0)

    return _result[["person_id"] + _fill_cols].reset_index(drop=True)
