"""social.py — per-user social / sharing feature group (stub).

compute_social(events_df, cutoff_map) -> pd.DataFrame
    Columns: person_id, is_social_user

AUDIT NOTE:
    The dataset was inspected for social/sharing events. The top-30
    events include only system and product-usage events (e.g. $identify,
    $set, agent_new_chat, run_block). No collaboration, invitation, or
    sharing events (e.g. invited_user, shared_notebook, referred_user)
    were found in this dataset.

    Rather than invent feature names that map to non-existent events,
    this module returns is_social_user = 0 for all users and emits a
    warning. This field is a placeholder to be populated if social
    events are added to the pipeline.
"""
from __future__ import annotations

import warnings

import pandas as pd


def compute_social(events_df: pd.DataFrame, cutoff_map: pd.Series) -> pd.DataFrame:
    """Compute social features, respecting per-user cutoff timestamps.

    Currently a stub: no social events identified in dataset.
    Returns is_social_user = 0 for all users in cutoff_map.

    Parameters
    ----------
    events_df : raw events DataFrame with columns [person_id, event, timestamp]
    cutoff_map : pd.Series indexed by person_id, values = cutoff_ts

    Returns
    -------
    pd.DataFrame with one row per person_id in cutoff_map
    """
    # ── Merge & cutoff filter (applied for consistency, even though unused) ─
    _df = events_df.merge(
        cutoff_map.rename("cutoff_ts").reset_index(),
        on="person_id",
        how="inner",
    )
    _df = _df[_df["timestamp"] <= _df["cutoff_ts"]]

    _n_kept = len(_df)
    _n_drop = len(events_df) - _n_kept
    print(
        f"compute_social: filtered {_n_drop:,} post-cutoff rows, "
        f"kept {_n_kept:,} for aggregation"
    )

    warnings.warn(
        "compute_social: no social/sharing events found in this dataset. "
        "is_social_user is set to 0 for all users. "
        "Populate this module when collaboration events become available.",
        UserWarning,
        stacklevel=2,
    )
    print(
        "  [WARNING] compute_social: no social events identified in dataset. "
        "is_social_user = 0 for all users (stub)."
    )

    # ── Return stub result aligned to spine ─────────────────────────────────
    _spine = cutoff_map.reset_index()[["person_id"]]
    _spine = _spine.copy()
    _spine["is_social_user"] = 0

    return _spine[["person_id", "is_social_user"]].reset_index(drop=True)
