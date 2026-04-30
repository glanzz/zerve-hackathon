"""
src/api/schemas.py — Pydantic v2 request/response models for the
User Intelligence API.

Feature order matches feature_names.pkl exactly (21 features):
  total_events, unique_event_types, total_sessions, avg_events_per_session,
  active_days, events_last_7d, events_last_14d, events_last_30d,
  events_last_14_to_7d, wow_growth, is_accelerating, days_since_last_event,
  days_since_first_event, account_age_days, feature_breadth, hit_aha_moment,
  ai_generation_count, run_block_count, agent_chat_count, friction_events,
  friction_ratio
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request ────────────────────────────────────────────────────────────────────

class UserFeatureInput(BaseModel):
    """All 21 features required by upgrade_predictor.pkl."""

    # Engagement
    total_events: float = Field(..., ge=0, description="Total event count for user")
    unique_event_types: float = Field(..., ge=0, description="Count of distinct event types fired")
    total_sessions: float = Field(..., ge=0, description="Distinct canvas sessions")
    avg_events_per_session: float = Field(..., ge=0, description="Mean events per canvas session")
    active_days: float = Field(..., ge=0, description="Count of distinct calendar days with activity")

    # Velocity
    events_last_7d: float = Field(..., ge=0, description="Events fired in the 7 days before cutoff")
    events_last_14d: float = Field(..., ge=0, description="Events fired in the 14 days before cutoff")
    events_last_30d: float = Field(..., ge=0, description="Events fired in the 30 days before cutoff")
    events_last_14_to_7d: float = Field(..., ge=0, description="Events in the 7–14 day window (events_last_14d - events_last_7d)")
    wow_growth: float = Field(..., description="Week-over-week growth ratio")
    is_accelerating: float = Field(..., ge=0, le=1, description="1 if wow_growth > 0.20 else 0")

    # Recency
    days_since_last_event: float = Field(..., ge=0, description="Days between last event and cutoff")
    days_since_first_event: float = Field(..., ge=0, description="Days between first event and cutoff")
    account_age_days: float = Field(..., ge=0, description="Days since new_user_created (or first event)")

    # Depth
    feature_breadth: float = Field(..., ge=0, description="Count of distinct core events fired")
    hit_aha_moment: float = Field(..., ge=0, le=1, description="1 if feature_breadth >= 2 else 0")
    ai_generation_count: float = Field(..., ge=0, description="Count of $ai_generation events")
    run_block_count: float = Field(..., ge=0, description="Count of run_block events")
    agent_chat_count: float = Field(..., ge=0, description="Count of agent_new_chat events")

    # Friction
    friction_events: float = Field(..., ge=0, description="Count of $exception events")
    friction_ratio: float = Field(..., ge=0, le=1, description="friction_events / total_events")

    model_config = {"json_schema_extra": {
        "example": {
            "total_events": 150, "unique_event_types": 12, "total_sessions": 3,
            "avg_events_per_session": 50.0, "active_days": 7,
            "events_last_7d": 80, "events_last_14d": 110, "events_last_30d": 140,
            "events_last_14_to_7d": 30, "wow_growth": 1.67, "is_accelerating": 1,
            "days_since_last_event": 0.5, "days_since_first_event": 20.0,
            "account_age_days": 20.0,
            "feature_breadth": 3, "hit_aha_moment": 1,
            "ai_generation_count": 45, "run_block_count": 20, "agent_chat_count": 5,
            "friction_events": 2, "friction_ratio": 0.013,
        }
    }}


class BatchPredictRequest(BaseModel):
    """Batch prediction request — up to 1000 users."""
    users: List[UserFeatureInput] = Field(..., min_length=1, max_length=1000)


# ── Response ───────────────────────────────────────────────────────────────────

class PredictionResponse(BaseModel):
    """Single-user prediction response."""
    person_id: Optional[str] = Field(None, description="Echo of input person_id (if provided)")
    upgrade_probability: float = Field(..., ge=0.0, le=1.0)
    upgrade_risk_tier: str = Field(..., description="HOT_LEAD | WARM_LEAD | NURTURE | COLD")
    recommended_action: str
    model_version: str
    inference_time_ms: float
    top_3_features: Dict[str, float] = Field(..., description="Top 3 feature names → values")


class BatchPredictResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    n_users: int
    inference_time_ms: float


class HealthResponse(BaseModel):
    """Health-check response."""
    status: str = "ok"
    model_loaded: bool
    oof_auc: Optional[float] = None
    oof_pr_auc: Optional[float] = None
    n_features: Optional[int] = None
    training_samples: Optional[int] = None


class ModelInfoResponse(BaseModel):
    """Full model metadata."""
    metadata: Dict
