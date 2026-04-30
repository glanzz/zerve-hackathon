"""
config/config.py — Datathon project configuration singleton.

Every other module:
    from config.config import CFG

CFG is a frozen dataclass. All paths, column names, leakage rules,
hyper-parameters, and signal-group lists live here.

Decisions (Sync 0 + empirical timing audit):
  - TENURE_ANCHOR_EVENT = 'new_user_created'  (sign_up fires late 92% of time)
  - credits_below_* / credits_exceeded BANNED: median delta 0.0-0.2h at upgrade
  - notebook_deployment_usage_tracked BANNED: negative median delta, post-upgrade
  - FRICTION_EVENTS = ['$exception'] only valid friction signal remaining
  - CORE_EVENTS_FOR_DEPTH: 6 empirically vetted aha-moment events
"""
from __future__ import annotations
import dataclasses
from typing import List


@dataclasses.dataclass(frozen=True)
class DatathonConfig:
    """Frozen singleton configuration for the entire pipeline."""

    # ── Paths ──────────────────────────────────────────────────────────────
    RAW_EVENTS_PATH: str = "data/raw/events.parquet"
    FEATURE_STORE_PATH: str = "data/processed/feature_store.parquet"
    LABELS_PATH: str = "data/processed/labels.parquet"
    SPLITS_PATH: str = "data/processed/splits.parquet"
    TRAIN_PATH: str = "data/processed/train.parquet"
    VAL_PATH: str = "data/processed/val.parquet"
    TEST_PATH: str = "data/processed/test.parquet"
    MODEL_DIR: str = "models/"
    SQL_DIR: str = "sql/"

    # ── Temporal ───────────────────────────────────────────────────────────
    FEATURE_WINDOW_DAYS: int = 30
    LEAKAGE_BUFFER_HOURS: int = 24

    # ── API ──────────────────────────────────────────────────────────
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    PREDICTION_HORIZON_DAYS: int = 7
    TRAIN_END_DATE: str = ""
    VAL_END_DATE: str = ""

    # ── Model ──────────────────────────────────────────────────────────────
    RANDOM_STATE: int = 42
    CV_FOLDS: int = 5
    TARGET_COL: str = "will_upgrade_in_7d"

    # ── Real data column names ─────────────────────────────────────────────
    USER_ID_COL: str = "person_id"
    TIMESTAMP_COL: str = "timestamp"
    EVENT_TYPE_COL: str = "event"
    UPGRADE_EVENT_NAME: str = "subscription_upgraded"

    # ── Tenure anchor ─────────────────────────────────────────────────────
    TENURE_ANCHOR_EVENT: str = "new_user_created"

    # ── Label inclusion threshold ─────────────────────────────────────────
    MIN_EVENTS_FOR_INCLUSION: int = 5

    # ── Banned event names ────────────────────────────────────────────────
    BANNED_EVENT_NAMES: List[str] = dataclasses.field(default_factory=lambda: [
        "subscription_upgraded",    # target — never use as feature
        "promo_code_redeemed",      # leakage: median 2.8h pre-upgrade
        "addon_credits_used",       # paid-tier-only, post-upgrade proxy
        "offer_declined",           # upgrade-funnel state reveal
        "credits_exceeded",         # fires AT upgrade moment (median 0.0h)
        "credits_below_1",          # same
        "credits_below_2",          # same
        "credits_below_3",          # same
        "credits_below_4",          # same
        "notebook_deployment_usage_tracked",  # post-upgrade (median -0.9h)
    ])

    # ── Banned feature column patterns ────────────────────────────────────
    BANNED_FEATURE_PATTERNS: List[str] = dataclasses.field(default_factory=lambda: [
        # CLAUDE.md base list
        "clicked_upgrade", "viewed_pricing", "started_trial",
        "billing_event", "upgrade_initiated", "conversion_flag",
        "account_type", "plan_type", "plan_name", "is_premium",
        "subscription_status", "post_", "_after_", "following_",
        "next_", "future_", "redeem_upgrade",
        # Sync 0 + timing audit additions
        "promo_code", "addon_credit", "credits_awarded",
        "credits_received", "total_addon_credits", "offer_declined",
        "notebook_deployment",
    ])

    # ── Core events for depth features ────────────────────────────────────
    CORE_EVENTS_FOR_DEPTH: List[str] = dataclasses.field(default_factory=lambda: [
        "agent_new_chat",
        "$ai_generation",
        "run_block",
        "agent_tool_call_create_block_tool",
        "agent_tool_call_run_block_tool",
        "agent_tool_call_get_block_tool",
    ])

    # ── Friction events ───────────────────────────────────────────────────
    FRICTION_EVENTS: List[str] = dataclasses.field(default_factory=lambda: [
        "$exception",
    ])


CFG = DatathonConfig()
