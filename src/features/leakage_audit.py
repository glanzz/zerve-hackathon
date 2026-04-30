"""leakage_audit.py — three-layer data-leakage audit for feature DataFrames.

Usage
-----
from src.features.leakage_audit import audit_features

report = audit_features(df, label_col="will_upgrade_in_7d")
# report is a list[dict] with keys: layer, column, reason, severity

Layers
------
1. Column-name pattern check
   Reject columns whose names contain any of:
     subscription_upgraded, upgrade, will_upgrade, promo_code,
     addon, notebook_deployment
   (case-insensitive, exact substring match)

2. High correlation check
   Pearson |r| > 0.95 with the label column → suspected leakage.

3. Post-cutoff data check
   If a 'timestamp' column exists, flag any rows where timestamp > user_cutoff_ts.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

# ── Layer 1: banned column-name substrings ────────────────────────────────────
_BANNED_NAME_PATTERNS: list[str] = [
    "subscription_upgraded",
    "upgrade",
    "will_upgrade",
    "promo_code",
    "addon",
    "notebook_deployment",
]

# Pre-compile as one case-insensitive pattern for speed
_BANNED_RE = re.compile(
    "|".join(re.escape(p) for p in _BANNED_NAME_PATTERNS),
    flags=re.IGNORECASE,
)


def _layer1_column_names(df: pd.DataFrame, label_col: str) -> list[dict]:
    """Flag feature columns whose names match banned patterns."""
    findings: list[dict] = []
    for col in df.columns:
        if col == label_col:
            continue
        match = _BANNED_RE.search(col)
        if match:
            findings.append(
                {
                    "layer": 1,
                    "column": col,
                    "reason": f"column name contains banned pattern '{match.group()}'",
                    "severity": "HIGH",
                }
            )
    return findings


def _layer2_high_correlation(
    df: pd.DataFrame,
    label_col: str,
    threshold: float = 0.95,
) -> list[dict]:
    """Flag numeric features with |Pearson r| > threshold against the label."""
    findings: list[dict] = []
    if label_col not in df.columns:
        return findings

    _label = pd.to_numeric(df[label_col], errors="coerce")
    _numeric_cols = df.select_dtypes(include="number").columns.tolist()

    for col in _numeric_cols:
        if col == label_col:
            continue
        _feat = pd.to_numeric(df[col], errors="coerce")
        _valid = _feat.notna() & _label.notna()
        if _valid.sum() < 10:
            continue
        try:
            _r = _feat[_valid].corr(_label[_valid])
        except Exception:
            continue
        if abs(_r) > threshold:
            findings.append(
                {
                    "layer": 2,
                    "column": col,
                    "reason": f"Pearson |r| = {abs(_r):.4f} > {threshold} with label",
                    "severity": "HIGH",
                }
            )
    return findings


def _layer3_post_cutoff_rows(df: pd.DataFrame) -> list[dict]:
    """Flag if any row has timestamp > user_cutoff_ts (temporal leakage)."""
    findings: list[dict] = []
    if "timestamp" not in df.columns or "user_cutoff_ts" not in df.columns:
        return findings

    _ts    = pd.to_datetime(df["timestamp"],      errors="coerce", utc=True)
    _cutoff = pd.to_datetime(df["user_cutoff_ts"], errors="coerce", utc=True)
    _leaky_mask = _ts > _cutoff
    _n = int(_leaky_mask.sum())
    if _n > 0:
        findings.append(
            {
                "layer": 3,
                "column": "timestamp / user_cutoff_ts",
                "reason": f"{_n} row(s) have timestamp > user_cutoff_ts (post-cutoff data)",
                "severity": "CRITICAL",
            }
        )
    return findings


def audit_features(
    df: pd.DataFrame,
    label_col: str = "will_upgrade_in_7d",
    correlation_threshold: float = 0.95,
) -> list[dict]:
    """Run three-layer leakage audit.

    Parameters
    ----------
    df : feature DataFrame (must include the label column)
    label_col : name of the binary label column
    correlation_threshold : |r| above which a numeric feature is flagged

    Returns
    -------
    list of finding dicts, each with keys:
        layer (int), column (str), reason (str), severity (str)
    """
    _findings: list[dict] = []
    _findings.extend(_layer1_column_names(df, label_col))
    _findings.extend(_layer2_high_correlation(df, label_col, correlation_threshold))
    _findings.extend(_layer3_post_cutoff_rows(df))
    return _findings
