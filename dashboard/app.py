
"""
dashboard/app.py — User Intelligence Dashboard (Streamlit, 4 tabs)

Tab 1  📊 Overview          — KPI cards + OOF prediction distribution
Tab 2  🗂️ Segment Explorer  — segment bar chart + conversion table
Tab 3  🎯 Live Predictor    — real-time /predict call to FastAPI
Tab 4  📈 Model Performance — fold scores, ROC/PR/calibration/SHAP

Run locally:
    streamlit run dashboard/app.py

All loaders use @st.cache_data. Graceful fallbacks when artifacts missing.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="User Intelligence Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paths ─────────────────────────────────────────────────────────────────────
_ROOT  = Path(__file__).parent.parent
_FS    = _ROOT / "data/processed/feature_store.parquet"
_LBLS  = _ROOT / "data/processed/labels.parquet"
_META  = _ROOT / "models/model_metadata.json"
_FOLDS = _ROOT / "models/fold_scores.csv"
_OOF   = _ROOT / "models/oof_predictions.csv"
_SHAP  = _ROOT / "models/shap_summary.png"
_ROC   = _ROOT / "models/evaluation/roc_curve.png"
_PR    = _ROOT / "models/evaluation/pr_curve.png"
_CAL   = _ROOT / "models/evaluation/calibration.png"
_CM    = _ROOT / "models/evaluation/confusion_matrix.png"
_HM    = _ROOT / "models/transition_heatmap.png"
_SEG   = _ROOT / "data/processed/segments.parquet"

# ── Zerve palette ─────────────────────────────────────────────────────────────
_BG  = "#1D1D20"
_FG  = "#fbfbff"
_SUB = "#909094"
_PAL = ["#A1C9F4","#FFB482","#8DE5A1","#FF9F9B","#D0BBFF","#ffd400","#17b26a"]
_HOT   = "#f04438"
_WARM  = "#FFB482"
_NURT  = "#ffd400"
_COLD  = "#A1C9F4"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    API_URL = st.text_input(
        "FastAPI Base URL",
        value="http://localhost:8000",
        help="Base URL of the User Intelligence API (no trailing slash)"
    )
    st.markdown("---")
    st.caption("Mission A — Upgrade Prediction")
    st.caption("Mission B — Funnel Segmentation")
    if st.button("🔄 Clear cache"):
        st.cache_data.clear()
        st.rerun()

# ── Cached loaders ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _load_fs():
    return pd.read_parquet(_FS) if _FS.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_meta():
    return json.loads(_META.read_text()) if _META.exists() else {}

@st.cache_data(show_spinner=False)
def _load_folds():
    return pd.read_csv(_FOLDS) if _FOLDS.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_oof():
    return pd.read_csv(_OOF) if _OOF.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_seg():
    return pd.read_parquet(_SEG) if _SEG.exists() else pd.DataFrame()

@st.cache_data(show_spinner=False)
def _load_lbls():
    return pd.read_parquet(_LBLS) if _LBLS.exists() else pd.DataFrame()

_fs    = _load_fs()
_meta  = _load_meta()
_folds = _load_folds()
_oof   = _load_oof()
_seg   = _load_seg()
_lbl   = _load_lbls()

# ── Helpers ───────────────────────────────────────────────────────────────────
def _show_png(path: Path, caption: str = "", width: int = 10, height: int = 5):
    if path.exists():
        _img = plt.imread(str(path))
        _fig, _ax = plt.subplots(figsize=(width, height))
        _fig.patch.set_facecolor(_BG)
        _ax.set_facecolor(_BG)
        _ax.imshow(_img)
        _ax.axis("off")
        if caption:
            _ax.set_title(caption, color=_FG, fontsize=11, pad=6)
        st.pyplot(_fig)
        plt.close(_fig)
    else:
        st.info(f"Artifact not found: `{path.relative_to(_ROOT)}`")

def _tier_colour(tier: str) -> str:
    t = tier.upper()
    if "HOT"  in t: return _HOT
    if "WARM" in t: return _WARM
    if "NURT" in t: return _NURT
    return _COLD

def _tier_emoji(tier: str) -> str:
    t = tier.upper()
    if "HOT"  in t: return "🔴"
    if "WARM" in t: return "🟠"
    if "NURT" in t: return "🟡"
    return "🔵"

# ── TABS ──────────────────────────────────────────────────────────────────────
_tab1, _tab2, _tab3, _tab4 = st.tabs([
    "📊 Overview",
    "🗂️ Segment Explorer",
    "🎯 Live Predictor",
    "📈 Model Performance",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═════════════════════════════════════════════════════════════════════════════
with _tab1:
    st.header("📊 Overview")

    _n_users  = len(_fs) if not _fs.empty else len(_lbl)
    _oof_auc  = _meta.get("oof_auc")
    _test_auc = _meta.get("test_auc")
    _pos_rate = (
        _fs["will_upgrade_in_7d"].mean() * 100
        if not _fs.empty and "will_upgrade_in_7d" in _fs.columns else None
    )
    _n_pos = (
        int(_fs["will_upgrade_in_7d"].sum())
        if not _fs.empty and "will_upgrade_in_7d" in _fs.columns else None
    )

    _c1, _c2, _c3, _c4, _c5 = st.columns(5)
    _c1.metric("Labeled Users",  f"{_n_users:,}"     if _n_users    else "—")
    _c2.metric("Upgraders",      f"{_n_pos:,}"        if _n_pos      else "—")
    _c3.metric("Upgrade Rate",   f"{_pos_rate:.2f}%"  if _pos_rate   else "—")
    _c4.metric("OOF AUC",        f"{_oof_auc:.4f}"    if _oof_auc    else "—")
    _c5.metric("Test AUC",       f"{_test_auc:.4f}"   if _test_auc   else "—")

    st.markdown("---")
    st.subheader("OOF Predicted Probability Distribution")
    if not _oof.empty and {"y_true", "y_pred_proba"}.issubset(_oof.columns):
        _hfig, _ax = plt.subplots(figsize=(10, 4))
        _hfig.patch.set_facecolor(_BG); _ax.set_facecolor(_BG)
        for _cls, _col, _lbl_str in [(0, _PAL[0], "Non-upgraders"), (1, _PAL[1], "Upgraders")]:
            _p = _oof.loc[_oof["y_true"] == _cls, "y_pred_proba"]
            _ax.hist(_p, bins=50, alpha=0.75, color=_col, label=_lbl_str, density=True)
        _ax.set_xlabel("Predicted upgrade probability", color=_FG, fontsize=11)
        _ax.set_ylabel("Density", color=_FG, fontsize=11)
        _ax.set_title("OOF Prediction Distribution by Class", color=_FG, fontsize=13, fontweight="bold")
        _ax.tick_params(colors=_SUB, labelsize=9)
        for _sp in _ax.spines.values(): _sp.set_edgecolor(_SUB)
        _ax.legend(facecolor=_BG, labelcolor=_FG, fontsize=9)
        st.pyplot(_hfig); plt.close(_hfig)

        _oof["risk_tier"] = pd.cut(
            _oof["y_pred_proba"], bins=[0, 0.20, 0.50, 0.80, 1.001],
            labels=["🔵 COLD","🟡 NURTURE","🟠 WARM_LEAD","🔴 HOT_LEAD"], right=False,
        )
        _td = (_oof.groupby("risk_tier", observed=True)
               .agg(users=("y_true","count"), actual_upgrades=("y_true","sum"))
               .assign(precision=lambda d: (d["actual_upgrades"]/d["users"]).round(3))
               .reset_index())
        _td.columns = ["Risk Tier","Users","Actual Upgraders","Precision"]
        st.dataframe(_td, use_container_width=True)
    else:
        st.info("Run training to populate OOF predictions.")

    if not _fs.empty:
        st.markdown("---")
        st.subheader("Feature Store — Descriptive Statistics")
        _sc = [c for c in ["total_events","unique_event_types","active_days",
                            "events_last_30d","feature_breadth","friction_ratio",
                            "ai_generation_count","run_block_count"] if c in _fs.columns]
        if _sc:
            _desc = _fs[_sc].describe().T[["mean","50%","min","max"]].round(2)
            _desc.columns = ["Mean","Median","Min","Max"]
            st.dataframe(_desc, use_container_width=True)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Segment Explorer
# ═════════════════════════════════════════════════════════════════════════════
with _tab2:
    st.header("🗂️ Segment Explorer")

    _seg_src = "segments.parquet"
    if _seg.empty and not _fs.empty and "funnel_segment" in _fs.columns:
        _seg = _fs.copy()
        _seg_src = "feature_store (funnel_segment column)"

    if not _seg.empty and "funnel_segment" in _seg.columns:
        st.caption(f"Source: `{_seg_src}`")
        _sc2  = "funnel_segment"
        _lc2  = "will_upgrade_in_7d" if "will_upgrade_in_7d" in _seg.columns else None
        _agg2 = {"user_count": (_sc2, "count")}
        if _lc2:
            _agg2["upgraders"]     = (_lc2, "sum")
            _agg2["conv_rate_pct"] = (_lc2, lambda x: round(x.mean() * 100, 2))
        _stats2 = (_seg.groupby(_sc2).agg(**_agg2).reset_index()
                   .sort_values("user_count", ascending=False))
        _stats2["share_pct"] = (_stats2["user_count"] / _stats2["user_count"].sum() * 100).round(1)
        _order2  = _stats2[_sc2].tolist()
        _colours2 = [_PAL[i % len(_PAL)] for i in range(len(_order2))]

        _bf, _ax2 = plt.subplots(figsize=(10, 4))
        _bf.patch.set_facecolor(_BG); _ax2.set_facecolor(_BG)
        _bars2 = _ax2.barh(_order2, _stats2["user_count"], color=_colours2, alpha=0.88)
        _ax2.set_xlabel("Users", color=_FG, fontsize=10)
        _ax2.set_title("Users per Segment", color=_FG, fontsize=13, fontweight="bold")
        _ax2.tick_params(colors=_FG, labelsize=9)
        for _sp in _ax2.spines.values(): _sp.set_edgecolor(_SUB)
        for _bar in _bars2:
            _w = _bar.get_width()
            _ax2.text(_w + _stats2["user_count"].max() * 0.01,
                      _bar.get_y() + _bar.get_height() / 2,
                      f"{int(_w):,}", va="center", color=_FG, fontsize=8)
        _bf.tight_layout(); st.pyplot(_bf); plt.close(_bf)

        if _lc2 and "conv_rate_pct" in _stats2.columns:
            _cf2, _axc = plt.subplots(figsize=(10, 4))
            _cf2.patch.set_facecolor(_BG); _axc.set_facecolor(_BG)
            _axc.barh(_order2, _stats2["conv_rate_pct"], color=_PAL[1], alpha=0.85)
            _axc.set_xlabel("Conversion Rate (%)", color=_FG, fontsize=10)
            _axc.set_title("7-Day Upgrade Conversion Rate by Segment", color=_FG, fontsize=13, fontweight="bold")
            _axc.tick_params(colors=_FG, labelsize=9)
            for _sp in _axc.spines.values(): _sp.set_edgecolor(_SUB)
            _cf2.tight_layout(); st.pyplot(_cf2); plt.close(_cf2)

        st.markdown("---")
        st.subheader("Segment Summary Table")
        _disp2 = _stats2.copy()
        _disp2.columns = [c.replace("_", " ").title() for c in _disp2.columns]
        st.dataframe(_disp2, use_container_width=True)
    else:
        st.info("Run `assign_segments()` (Step 22) to populate segment data.")

    st.markdown("---")
    st.subheader("🔀 Segment Transition Heatmap")
    _show_png(_HM, "Weekly Segment Transition Probabilities", width=11, height=6)

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Live Predictor
# ═════════════════════════════════════════════════════════════════════════════
with _tab3:
    st.header("🎯 Live Predictor")
    st.markdown(
        "Enter user engagement features below and call the **FastAPI `/predict` endpoint** "
        "for a real-time upgrade probability score."
    )

    # ── API health banner ──────────────────────────────────────────────────────
    _api_status_placeholder = st.empty()
    try:
        _hr = requests.get(f"{API_URL}/health", timeout=3)
        if _hr.status_code == 200:
            _hdata = _hr.json()
            _api_status_placeholder.success(
                f"✅ API online — OOF AUC: **{_hdata.get('oof_auc','n/a')}** | "
                f"Features: **{_hdata.get('n_features','n/a')}** | "
                f"Model loaded: **{_hdata.get('model_loaded','?')}**"
            )
        else:
            _api_status_placeholder.warning(f"⚠️ API returned HTTP {_hr.status_code}")
    except Exception:
        _api_status_placeholder.warning(
            f"⚠️ API unreachable at `{API_URL}` — update the URL in the sidebar, "
            "or deploy the FastAPI service first."
        )

    st.markdown("---")

    # ── Feature inputs ─────────────────────────────────────────────────────────
    st.subheader("Feature Inputs")

    # Derive sensible defaults from feature store percentiles
    def _pct(col, q):
        if not _fs.empty and col in _fs.columns:
            return float(_fs[col].quantile(q))
        return 0.0

    _col_a, _col_b = st.columns(2)

    with _col_a:
        st.markdown("**📦 Engagement**")
        inp_total_events          = st.number_input("total_events",          min_value=0.0, value=_pct("total_events", 0.5),          step=1.0, format="%.0f")
        inp_unique_event_types    = st.number_input("unique_event_types",    min_value=0.0, value=_pct("unique_event_types", 0.5),    step=1.0, format="%.0f")
        inp_total_sessions        = st.number_input("total_sessions",        min_value=0.0, value=_pct("total_sessions", 0.5),        step=1.0, format="%.0f")
        inp_avg_events_per_session= st.number_input("avg_events_per_session",min_value=0.0, value=max(_pct("avg_events_per_session",0.5),0.0), step=0.1)
        inp_active_days           = st.number_input("active_days",           min_value=0.0, value=_pct("active_days", 0.5),           step=1.0, format="%.0f")

        st.markdown("**⏱ Recency**")
        inp_days_since_last_event  = st.number_input("days_since_last_event",  min_value=0.0, value=_pct("days_since_last_event", 0.5),  step=0.5)
        inp_days_since_first_event = st.number_input("days_since_first_event", min_value=0.0, value=_pct("days_since_first_event", 0.5), step=1.0)
        inp_account_age_days       = st.number_input("account_age_days",       min_value=0.0, value=_pct("account_age_days", 0.5),       step=1.0)

        st.markdown("**💥 Friction**")
        inp_friction_events = st.number_input("friction_events", min_value=0.0, value=_pct("friction_events", 0.5), step=1.0, format="%.0f")
        _auto_ratio = inp_friction_events / max(inp_total_events, 1)
        inp_friction_ratio  = st.number_input("friction_ratio",  min_value=0.0, max_value=1.0, value=round(_auto_ratio, 4), step=0.001, format="%.4f", help="Auto-computed as friction_events / total_events")

    with _col_b:
        st.markdown("**🚀 Velocity**")
        inp_events_last_7d      = st.number_input("events_last_7d",      min_value=0.0, value=_pct("events_last_7d", 0.5),      step=1.0, format="%.0f")
        inp_events_last_14d     = st.number_input("events_last_14d",     min_value=0.0, value=_pct("events_last_14d", 0.5),     step=1.0, format="%.0f")
        inp_events_last_30d     = st.number_input("events_last_30d",     min_value=0.0, value=_pct("events_last_30d", 0.5),     step=1.0, format="%.0f")
        inp_events_last_14_to_7d= st.number_input("events_last_14_to_7d",min_value=0.0, value=max(inp_events_last_14d - inp_events_last_7d, 0.0), step=1.0, format="%.0f", help="events 7–14 days before cutoff")
        _auto_wow = (inp_events_last_7d / max(inp_events_last_14_to_7d, 1)) - 1
        inp_wow_growth          = st.number_input("wow_growth",          value=round(_auto_wow, 4), step=0.01, help="Week-over-week growth ratio (auto-computed)")
        inp_is_accelerating     = st.selectbox("is_accelerating", options=[0, 1], index=int(inp_wow_growth > 0.20), help="1 if wow_growth > 0.20")

        st.markdown("**🔬 Depth**")
        inp_feature_breadth     = st.number_input("feature_breadth",     min_value=0.0, value=_pct("feature_breadth", 0.5),     step=1.0, format="%.0f")
        inp_hit_aha_moment      = st.selectbox("hit_aha_moment", options=[0, 1], index=int(inp_feature_breadth >= 2), help="1 if feature_breadth ≥ 2")
        inp_ai_generation_count = st.number_input("ai_generation_count", min_value=0.0, value=_pct("ai_generation_count", 0.5), step=1.0, format="%.0f")
        inp_run_block_count     = st.number_input("run_block_count",     min_value=0.0, value=_pct("run_block_count", 0.5),     step=1.0, format="%.0f")
        inp_agent_chat_count    = st.number_input("agent_chat_count",    min_value=0.0, value=_pct("agent_chat_count", 0.5),    step=1.0, format="%.0f")

    # Assemble payload
    _payload = {
        "total_events":           inp_total_events,
        "unique_event_types":     inp_unique_event_types,
        "total_sessions":         inp_total_sessions,
        "avg_events_per_session": inp_avg_events_per_session,
        "active_days":            inp_active_days,
        "events_last_7d":         inp_events_last_7d,
        "events_last_14d":        inp_events_last_14d,
        "events_last_30d":        inp_events_last_30d,
        "events_last_14_to_7d":   inp_events_last_14_to_7d,
        "wow_growth":             inp_wow_growth,
        "is_accelerating":        float(inp_is_accelerating),
        "days_since_last_event":  inp_days_since_last_event,
        "days_since_first_event": inp_days_since_first_event,
        "account_age_days":       inp_account_age_days,
        "feature_breadth":        inp_feature_breadth,
        "hit_aha_moment":         float(inp_hit_aha_moment),
        "ai_generation_count":    inp_ai_generation_count,
        "run_block_count":        inp_run_block_count,
        "agent_chat_count":       inp_agent_chat_count,
        "friction_events":        inp_friction_events,
        "friction_ratio":         inp_friction_ratio,
    }

    st.markdown("---")
    st.subheader("JSON Payload Preview")
    st.json(_payload)

    # ── Predict button ──────────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("🔮 Predict Upgrade Probability", type="primary", use_container_width=True):
        with st.spinner("Calling /predict …"):
            try:
                _resp = requests.post(
                    f"{API_URL}/predict",
                    json=_payload,
                    timeout=10,
                )
                if _resp.status_code == 200:
                    _pred = _resp.json()
                    _prob = _pred.get("upgrade_probability", 0.0)
                    _tier = _pred.get("upgrade_risk_tier", "COLD")
                    _action = _pred.get("recommended_action", "—")
                    _latency = _pred.get("inference_time_ms", None)
                    _top3    = _pred.get("top_3_features", {})

                    # Result cards
                    _r1, _r2, _r3 = st.columns(3)
                    _r1.metric(
                        "Upgrade Probability",
                        f"{_prob:.1%}",
                        delta=f"{_prob - 0.0209:+.1%} vs base rate",
                    )
                    _r2.metric(
                        "Risk Tier",
                        f"{_tier_emoji(_tier)} {_tier}",
                    )
                    _r3.metric(
                        "Inference",
                        f"{_latency:.1f} ms" if _latency else "—",
                    )

                    # Probability gauge bar
                    _gfig, _gax = plt.subplots(figsize=(10, 1.2))
                    _gfig.patch.set_facecolor(_BG); _gax.set_facecolor(_BG)
                    # Background track
                    _gax.barh(0, 1.0, color="#333337", height=0.5)
                    # Filled portion
                    _fill_col = _HOT if _prob >= 0.80 else (_WARM if _prob >= 0.50 else (_NURT if _prob >= 0.20 else _COLD))
                    _gax.barh(0, _prob, color=_fill_col, height=0.5, alpha=0.92)
                    # Tick marks
                    for _t in [0.20, 0.50, 0.80]:
                        _gax.axvline(_t, color=_SUB, linewidth=1, linestyle="--", alpha=0.7)
                    _gax.set_xlim(0, 1)
                    _gax.set_yticks([])
                    _tick_vals  = [0, 0.20, 0.50, 0.80, 1.0]
                    _tick_lbls  = ["0%", "20%", "50%", "80%", "100%"]
                    _gax.set_xticks(_tick_vals)
                    _gax.set_xticklabels(_tick_lbls, color=_FG, fontsize=9)
                    for _sp in _gax.spines.values(): _sp.set_visible(False)
                    _gax.set_title(f"Upgrade Probability: {_prob:.1%}  ({_tier})",
                                   color=_FG, fontsize=12, fontweight="bold", pad=8)
                    _gfig.tight_layout()
                    st.pyplot(_gfig); plt.close(_gfig)

                    # Recommended action callout
                    st.markdown(f"""
> **Recommended action:** {_action}
                    """)

                    # Top-3 features
                    if _top3:
                        st.markdown("**Top contributing features:**")
                        _t3df = pd.DataFrame(
                            list(_top3.items()), columns=["Feature", "Value"]
                        ).sort_values("Value", ascending=False)
                        st.dataframe(_t3df, use_container_width=True, hide_index=True)

                    # Raw JSON expander
                    with st.expander("Raw API response"):
                        st.json(_pred)

                else:
                    st.error(f"API returned HTTP {_resp.status_code}: {_resp.text[:500]}")

            except requests.exceptions.ConnectionError:
                st.error(
                    f"❌ Could not connect to `{API_URL}/predict`. "
                    "Ensure the FastAPI service is running and the URL is correct."
                )
            except requests.exceptions.Timeout:
                st.error("❌ Request timed out (10s). The model server may be overloaded.")
            except Exception as _ex:
                st.error(f"❌ Unexpected error: {_ex}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 4 — Model Performance
# ═════════════════════════════════════════════════════════════════════════════
with _tab4:
    st.header("📈 Model Performance")

    # ── Top KPIs from metadata ─────────────────────────────────────────────────
    if _meta:
        _m1, _m2, _m3, _m4, _m5, _m6 = st.columns(6)
        _m1.metric("OOF AUC",       f"{_meta.get('oof_auc',0):.4f}")
        _m2.metric("Test AUC",      f"{_meta.get('test_auc',0):.4f}")
        _m3.metric("OOF PR-AUC",    f"{_meta.get('oof_pr_auc', _meta.get('oof_pr-auc',0)):.4f}" if _meta.get('oof_pr_auc') or _meta.get('oof_pr-auc') else "—")
        _m4.metric("N Features",    str(_meta.get("n_features","—")))
        _m5.metric("Train samples", f"{_meta.get('n_train',0):,}")
        _m6.metric("Trained at",    str(_meta.get("trained_at","—"))[:10])
    else:
        st.info("model_metadata.json not found — run training (Step 20).")

    st.markdown("---")

    # ── Fold scores ────────────────────────────────────────────────────────────
    st.subheader("Cross-Validation Fold Scores")
    if not _folds.empty:
        # Normalise column names
        _fd = _folds.copy()
        _fd.columns = [c.strip().lower().replace(" ", "_") for c in _fd.columns]

        # Detect AUC column
        _auc_col = next((c for c in _fd.columns if "auc" in c), None)
        _fold_col = next((c for c in _fd.columns if "fold" in c), None)

        # Pretty display table
        _fd_disp = _fd.copy()
        _fd_disp.columns = [c.replace("_", " ").title() for c in _fd_disp.columns]

        # Convert Fold column to string to avoid mixed-type issues with Arrow serialization
        if _fold_col:
            _fold_display_col = _fold_col.replace("_", " ").title()
            if _fold_display_col in _fd_disp.columns:
                _fd_disp[_fold_display_col] = _fd_disp[_fold_display_col].astype(str)

        if _meta.get("oof_auc"):
            _mean_row = {c: "—" for c in _fd_disp.columns}
            if _fold_col:
                _mean_row[_fold_col.replace("_"," ").title()] = "Mean"
            if _auc_col:
                _mean_row[_auc_col.replace("_"," ").title()] = f"{_fd[_auc_col].mean():.4f}"
            _fd_disp = pd.concat([_fd_disp, pd.DataFrame([_mean_row])], ignore_index=True)

        st.dataframe(_fd_disp, use_container_width=True, hide_index=True)

        # Per-fold AUC bar chart
        if _auc_col and _fold_col:
            _ff, _fax = plt.subplots(figsize=(8, 3.5))
            _ff.patch.set_facecolor(_BG); _fax.set_facecolor(_BG)
            _fold_labels = [str(int(v)) if str(v).replace(".","").isdigit() else str(v)
                            for v in _fd[_fold_col]]
            _bars_f = _fax.bar(_fold_labels, _fd[_auc_col], color=_PAL[0], alpha=0.85, width=0.55)
            _mean_val = _fd[_auc_col].mean()
            _fax.axhline(_mean_val, color=_PAL[1], linewidth=1.8, linestyle="--",
                         label=f"Mean = {_mean_val:.4f}")
            # Y ticks — pre-computed strings (no callable formatters)
            _y_min = max(0, _fd[_auc_col].min() - 0.02)
            _y_max = min(1.0, _fd[_auc_col].max() + 0.02)
            _ytick_vals = np.linspace(_y_min, _y_max, 6)
            _ytick_lbls = [f"{v:.3f}" for v in _ytick_vals]
            _fax.set_yticks(_ytick_vals)
            _fax.set_yticklabels(_ytick_lbls, color=_FG, fontsize=8)
            _fax.set_ylim(_y_min, _y_max)
            # Annotate bars
            for _b, _v in zip(_bars_f, _fd[_auc_col]):
                _fax.text(_b.get_x() + _b.get_width() / 2, _v + 0.002,
                          f"{_v:.4f}", ha="center", va="bottom", color=_FG, fontsize=8)
            _fax.set_xlabel("Fold", color=_FG, fontsize=10)
            _fax.set_ylabel("ROC-AUC", color=_FG, fontsize=10)
            _fax.set_title("Per-Fold Cross-Validation AUC", color=_FG, fontsize=13, fontweight="bold")
            _fax.tick_params(axis="x", colors=_FG, labelsize=9)
            for _sp in _fax.spines.values(): _sp.set_edgecolor(_SUB)
            _fax.legend(facecolor=_BG, labelcolor=_FG, fontsize=9)
            _ff.tight_layout(); st.pyplot(_ff); plt.close(_ff)
    else:
        st.info("fold_scores.csv not found at `models/fold_scores.csv` — run training first.")

    st.markdown("---")

    # ── SHAP summary ───────────────────────────────────────────────────────────
    st.subheader("SHAP Feature Importance")
    _show_png(_SHAP, "SHAP Summary Plot — Top Feature Importances", width=11, height=7)

    st.markdown("---")

    # ── Evaluation plots (2 × 2 grid) ─────────────────────────────────────────
    st.subheader("Evaluation Curves")
    _e1, _e2 = st.columns(2)
    with _e1:
        st.markdown("**ROC Curve**")
        _show_png(_ROC, width=6, height=5)
        st.markdown("**Calibration Plot**")
        _show_png(_CAL, width=6, height=5)
    with _e2:
        st.markdown("**Precision-Recall Curve**")
        _show_png(_PR, width=6, height=5)
        st.markdown("**Confusion Matrix**")
        _show_png(_CM, width=6, height=5)

    st.markdown("---")
    st.subheader("Segment Transition Heatmap")
    _show_png(_HM, "Weekly Segment Transition Probabilities", width=11, height=6)
