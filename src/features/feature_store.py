"""feature_store.py — orchestrator for all feature groups.

Usage
-----
    python -m src.features.feature_store
    # or import and call build_feature_store() directly

Output
------
    CFG.FEATURE_STORE_PATH  (data/processed/feature_store.parquet)

Steps
-----
1. Load raw events via loader.load_raw_events()
2. Load labels from CFG.LABELS_PATH (error if missing — run
   sql/02_label_generation.sql first)
3. Sanitize temporal columns (guarantee tz-aware UTC) ONCE here,
   before passing anything to compute_* functions
4. Build cutoff_map from the sanitized labels
5. Call all 6 compute_* functions with (events_df, cutoff_map)
6. Left-join all feature groups onto labels spine
7. Fill NaN with 0
8. Save to CFG.FEATURE_STORE_PATH
9. Print shape, column count, column list, head(5)

NOTE: leakage_audit is NOT run here — that is Step 17b.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# ── path bootstrap so this file can be run as __main__ ─────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from config.config import CFG
from src.data.loader import load_raw_events
from src.features.engagement  import compute_engagement
from src.features.velocity    import compute_velocity
from src.features.recency     import compute_recency
from src.features.depth       import compute_depth
from src.features.friction    import compute_friction
from src.features.social      import compute_social


def build_feature_store(
    events_df: pd.DataFrame | None = None,
    labels_df: pd.DataFrame | None = None,
    save: bool = True,
) -> pd.DataFrame:
    """Run all feature groups and join onto labels spine.

    Parameters
    ----------
    events_df : pd.DataFrame, optional
        Pre-loaded events. If None, loads via load_raw_events().
    labels_df : pd.DataFrame, optional
        Pre-loaded labels. If None, loads from CFG.LABELS_PATH.
    save : bool
        Whether to save the result to CFG.FEATURE_STORE_PATH.

    Returns
    -------
    pd.DataFrame — full feature store (not yet audited)
    """
    # ── 1. Load events ──────────────────────────────────────────────────────
    if events_df is None:
        print("Loading raw events...")
        events_df = load_raw_events()
    print(f"  {len(events_df):,} rows, {events_df['person_id'].nunique():,} users")

    # ── 2. Load labels ──────────────────────────────────────────────────────
    if labels_df is None:
        _labels_path = Path(CFG.LABELS_PATH)
        if not _labels_path.exists():
            raise FileNotFoundError(
                f"Labels file not found at '{_labels_path}'. "
                "Run sql/02_label_generation.sql first to generate labels."
            )
        print(f"Loading labels from {_labels_path}...")
        labels_df = pd.read_parquet(_labels_path)
    print(f"  {len(labels_df):,} labeled users  "
          f"(positives: {labels_df[CFG.TARGET_COL].sum():,})")

    # ── Type-coerce all temporal columns ONCE before passing to compute_* ──
    events_df = events_df.copy()
    events_df['timestamp'] = pd.to_datetime(
        events_df['timestamp'], utc=True, errors='coerce'
    )
    _n_bad = events_df['timestamp'].isna().sum()
    if _n_bad:
        print(f"  [WARNING] dropping {_n_bad:,} rows with unparseable timestamps")
        events_df = events_df.dropna(subset=['timestamp'])

    labels_df = labels_df.copy()
    labels_df['user_cutoff_ts'] = pd.to_datetime(
        labels_df['user_cutoff_ts'], utc=True, errors='coerce'
    )
    _n_bad_lab = labels_df['user_cutoff_ts'].isna().sum()
    if _n_bad_lab:
        raise ValueError(
            f"{_n_bad_lab} labels have unparseable user_cutoff_ts; aborting"
        )

    # ── 3. Build cutoff_map ─────────────────────────────────────────────────
    cutoff_map = (
        labels_df[[CFG.USER_ID_COL, 'user_cutoff_ts']]
        .set_index(CFG.USER_ID_COL)['user_cutoff_ts']
    )

    print(f"build_feature_store: events {len(events_df):,} rows, "
          f"labels {len(labels_df):,} users, cutoff_map {len(cutoff_map):,}")

    # ── 4. Compute all feature groups ───────────────────────────────────────
    print()
    print("Computing feature groups...")

    _features = [
        ("engagement",  compute_engagement(events_df, cutoff_map)),
        ("velocity",    compute_velocity(events_df, cutoff_map)),
        ("recency",     compute_recency(events_df, cutoff_map)),
        ("depth",       compute_depth(events_df, cutoff_map)),
        ("friction",    compute_friction(events_df, cutoff_map)),
        ("social",      compute_social(events_df, cutoff_map)),
    ]

    for _name, _df in _features:
        print(f"  {_name}: {_df.shape} — {list(_df.columns)}")

    # ── 5. Left-join all groups onto labels spine ────────────────────────────
    print()
    print("Joining feature groups onto labels spine...")
    feature_store = labels_df.copy()
    for _name, _df in _features:
        feature_store = feature_store.merge(_df, on=CFG.USER_ID_COL, how='left')
        print(f"  after {_name}: shape={feature_store.shape}")

    # ── 6. Fill NaN with 0 ──────────────────────────────────────────────────
    _before_nulls = feature_store.isnull().sum().sum()
    _non_label_cols = [c for c in feature_store.columns
                       if c not in [CFG.USER_ID_COL, CFG.TARGET_COL,
                                    'user_cutoff_ts', 'split']]
    feature_store[_non_label_cols] = feature_store[_non_label_cols].fillna(0)
    _after_nulls = feature_store.isnull().sum().sum()
    print(f"  NaN fill: {_before_nulls:,} → {_after_nulls:,} nulls")

    # ── 7-8. Save ───────────────────────────────────────────────────────────
    if save:
        _out = Path(CFG.FEATURE_STORE_PATH)
        _out.parent.mkdir(parents=True, exist_ok=True)
        feature_store.to_parquet(_out, index=False)
        print(f"\nSaved: {_out}")

    # ── 9. Print summary ────────────────────────────────────────────────────
    print(f"\nShape       : {feature_store.shape}")
    print(f"Column count: {feature_store.shape[1]}")
    print(f"Columns     : {list(feature_store.columns)}")
    print(f"\nHead(5):")
    print(feature_store.head(5).to_string(index=False))

    return feature_store


if __name__ == "__main__":
    build_feature_store()
