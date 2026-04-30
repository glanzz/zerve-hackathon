"""
segments.py — Rule-based funnel segment assignment for Mission B.

Segment philosophy (derived from EDA on real Zerve event data):
  - Every user must fall into exactly one stage (complete coverage).
  - Rules are deterministic, based only on feature_store columns.
  - Stages are ordered by engagement depth; np.select uses first-match.
  - Unclassified fallback must be < 10% (raises ValueError if exceeded).

7-stage funnel grounded in real Zerve behavior (from 00_ingest_profile.sql EDA):
  Top events: credits_used, $ai_generation, $exception, addon_credits_used,
              notebook_deployment_usage_tracked, $web_vitals,
              agent_tool_call_* series, block execution events.
  Funnel: Power User → AI Builder → Collaborator → Explorer → Creator → Onboarding → Dormant
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Any, List


# ── Segment definitions ─────────────────────────────────────────────────────
# Order matters: highest engagement first (np.select first-match semantics)
SEGMENT_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    "Power User": {
        "description": (
            "Heaviest platform consumers: multi-feature, high AI generation, "
            "agent tool invocations, block execution, hit AHA moment. "
            "Broadest feature footprint — most likely to upgrade."
        ),
        "rules": (
            "hit_aha_moment==1 AND feature_breadth>=4 AND "
            "ai_generation_count>=10 AND (run_block_count+agent_chat_count)>=5 "
            "AND unique_event_types>=20"
        ),
        "action": "Offer upgrade prompt, personalised plan comparison, priority support",
        "upgrade_likelihood": "High",
    },
    "AI Builder": {
        "description": (
            "Users actively using AI generation and block execution. "
            "Have hit the AHA moment but slightly narrower feature breadth than Power Users."
        ),
        "rules": (
            "hit_aha_moment==1 AND ai_generation_count>=5 AND "
            "run_block_count>=1 AND unique_event_types>=10"
        ),
        "action": "Showcase advanced AI features and collaboration tools",
        "upgrade_likelihood": "High",
    },
    "Collaborator": {
        "description": (
            "Users engaged in agent conversations and multi-feature workflows. "
            "Consistent activity across multiple days."
        ),
        "rules": (
            "agent_chat_count>=3 AND feature_breadth>=2 AND "
            "active_days>=3 AND unique_event_types>=8"
        ),
        "action": "Highlight team collaboration features and deployment capabilities",
        "upgrade_likelihood": "Medium",
    },
    "Explorer": {
        "description": (
            "Users experimenting across multiple feature types. "
            "Good event volume, beginning to see breadth — AHA not yet triggered."
        ),
        "rules": (
            "feature_breadth>=2 AND total_events>=20 AND "
            "unique_event_types>=8 AND days_since_last_event<=14"
        ),
        "action": "Guide toward AHA moment — suggest AI features and block execution",
        "upgrade_likelihood": "Medium",
    },
    "Creator": {
        "description": (
            "Users who have created content (run blocks or generated AI output). "
            "Some platform depth but limited breadth or recency."
        ),
        "rules": (
            "(run_block_count>=1 OR ai_generation_count>=1) AND "
            "total_events>=5 AND unique_event_types>=3"
        ),
        "action": "Nurture with tutorials and use-case showcases to deepen engagement",
        "upgrade_likelihood": "Low",
    },
    "Onboarding": {
        "description": (
            "New or very early-stage users. Low event counts, "
            "narrow feature exposure, joined recently or hasn't progressed yet."
        ),
        "rules": (
            "total_events<30 AND feature_breadth<=1 AND "
            "days_since_first_event<=30"
        ),
        "action": "Onboarding flow, first-run prompts, activation nudges",
        "upgrade_likelihood": "Low",
    },
    "Dormant": {
        "description": (
            "Users who have not been active recently or have very low lifetime "
            "engagement. Risk of permanent churn."
        ),
        "rules": (
            "days_since_last_event>14 OR "
            "(total_events<5 AND days_since_first_event>14)"
        ),
        "action": "Win-back campaign — re-engagement email with new feature highlight",
        "upgrade_likelihood": "Low",
    },
}

# Ordered list for np.select (highest-priority first)
SEGMENT_ORDER: List[str] = [
    "Power User",
    "AI Builder",
    "Collaborator",
    "Explorer",
    "Creator",
    "Onboarding",
    "Dormant",
]


def _build_conditions(df: pd.DataFrame) -> list:
    """
    Build boolean condition arrays for each segment in SEGMENT_ORDER.

    Args:
        df: feature_store DataFrame with all required columns present.

    Returns:
        List of boolean Series, one per segment in SEGMENT_ORDER.
    """
    return [
        # Power User — broadest engagement + AHA + heavy AI + agent/block use
        (
            (df["hit_aha_moment"] == 1)
            & (df["feature_breadth"] >= 4)
            & (df["ai_generation_count"] >= 10)
            & ((df["run_block_count"] + df["agent_chat_count"]) >= 5)
            & (df["unique_event_types"] >= 20)
        ),
        # AI Builder — AHA hit + meaningful AI gen + block execution
        (
            (df["hit_aha_moment"] == 1)
            & (df["ai_generation_count"] >= 5)
            & (df["run_block_count"] >= 1)
            & (df["unique_event_types"] >= 10)
        ),
        # Collaborator — agent conversations + multi-feature + recurring activity
        (
            (df["agent_chat_count"] >= 3)
            & (df["feature_breadth"] >= 2)
            & (df["active_days"] >= 3)
            & (df["unique_event_types"] >= 8)
        ),
        # Explorer — multi-feature breadth + moderate volume + recently active (14d window)
        (
            (df["feature_breadth"] >= 2)
            & (df["total_events"] >= 20)
            & (df["unique_event_types"] >= 8)
            & (df["days_since_last_event"] <= 14)
        ),
        # Creator — any content creation (blocks or AI), minimal event floor lowered to 5
        (
            ((df["run_block_count"] >= 1) | (df["ai_generation_count"] >= 1))
            & (df["total_events"] >= 5)
            & (df["unique_event_types"] >= 3)
        ),
        # Onboarding — low volume, narrow breadth, within first 30 days
        (
            (df["total_events"] < 30)
            & (df["feature_breadth"] <= 1)
            & (df["days_since_first_event"] <= 30)
        ),
        # Dormant — inactive >14d OR very minimal engagement overall
        (
            (df["days_since_last_event"] > 14)
            | ((df["total_events"] < 5) & (df["days_since_first_event"] > 14))
        ),
    ]


def assign_segments(feature_store: pd.DataFrame) -> pd.DataFrame:
    """
    Apply rule-based segment assignment via np.select.

    Adds a 'funnel_segment' column to a copy of feature_store.
    Unclassified users receive the label 'Unclassified'.

    Args:
        feature_store: DataFrame with feature_store schema (person_id +
                       all feature columns). Must contain the columns
                       referenced in SEGMENT_DEFINITIONS.

    Returns:
        DataFrame with 'funnel_segment' column added.

    Raises:
        ValueError: If more than 10% of users are Unclassified.
    """
    required_cols = [
        "hit_aha_moment", "feature_breadth", "ai_generation_count",
        "run_block_count", "agent_chat_count", "unique_event_types",
        "total_events", "active_days", "days_since_last_event",
        "days_since_first_event",
    ]
    missing = [c for c in required_cols if c not in feature_store.columns]
    if missing:
        raise KeyError(f"assign_segments: missing required columns: {missing}")

    df = feature_store.copy()
    conditions = _build_conditions(df)

    df["funnel_segment"] = np.select(conditions, SEGMENT_ORDER, default="Unclassified")

    n_total      = len(df)
    n_unclassified = (df["funnel_segment"] == "Unclassified").sum()
    pct_unclassified = n_unclassified / n_total * 100

    dist = df["funnel_segment"].value_counts()
    print(f"assign_segments: {n_total:,} users assigned.")
    print(f"  Segment distribution:")
    for seg in SEGMENT_ORDER + ["Unclassified"]:
        cnt = dist.get(seg, 0)
        pct = cnt / n_total * 100
        bar = "█" * int(pct / 2)
        print(f"    {seg:<20} {cnt:>6,}  ({pct:5.1f}%)  {bar}")
    print()

    if pct_unclassified > 10.0:
        raise ValueError(
            f"assign_segments: {pct_unclassified:.1f}% Unclassified "
            f"({n_unclassified:,}/{n_total:,}) — exceeds 10% threshold. "
            "Tune segment rules to reduce fallback coverage."
        )
    else:
        print(
            f"  ✅ Unclassified check PASSED: {pct_unclassified:.1f}% "
            f"({n_unclassified:,} users) < 10% threshold."
        )

    return df


def get_segment_stats(df_with_segments: pd.DataFrame) -> pd.DataFrame:
    """
    Compute summary statistics per funnel segment.

    For each segment computes: count, pct_of_total,
    avg upgrade probability (if present), avg active_days,
    avg feature_breadth, avg friction_ratio.

    Args:
        df_with_segments: Output of assign_segments() — must have
                          'funnel_segment' column.

    Returns:
        Summary DataFrame indexed by funnel_segment.
    """
    if "funnel_segment" not in df_with_segments.columns:
        raise KeyError(
            "get_segment_stats: 'funnel_segment' column missing — run assign_segments first."
        )

    n_total = len(df_with_segments)
    agg_cols = {
        "person_id": "count",
        "active_days": "mean",
        "feature_breadth": "mean",
        "friction_ratio": "mean",
        "unique_event_types": "mean",
        "total_events": "mean",
        "hit_aha_moment": "mean",
        "ai_generation_count": "mean",
    }
    for opt in ["upgrade_probability", "will_upgrade_in_7d"]:
        if opt in df_with_segments.columns:
            agg_cols[opt] = "mean"

    stats = df_with_segments.groupby("funnel_segment").agg(agg_cols).reset_index()
    stats.rename(columns={"person_id": "user_count"}, inplace=True)
    stats.insert(2, "pct_of_total", (stats["user_count"] / n_total * 100).round(1))

    order_map = {s: i for i, s in enumerate(SEGMENT_ORDER + ["Unclassified"])}
    stats["_order"] = stats["funnel_segment"].map(order_map).fillna(99)
    stats = stats.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    for col in stats.select_dtypes("float").columns:
        stats[col] = stats[col].round(3)

    return stats


if __name__ == "__main__":
    import pandas as pd
    import sys, os
    sys.path.insert(0, os.getcwd())
    from config.config import CFG

    print("Loading feature store...")
    fs = pd.read_parquet(CFG.FEATURE_STORE_PATH)
    print(f"  Shape: {fs.shape}")

    print("\nRunning assign_segments()...")
    result = assign_segments(fs)
    print("\nget_segment_stats():")
    print(get_segment_stats(result).to_string())
