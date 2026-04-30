# CLAUDE.md — Datathon Agent Briefing
## Complete Build Specification for Claude Code

> **You are a senior ML engineer and your job is to build this entire project from scratch,
> file by file, in the correct dependency order. Read this document fully before writing
> a single line of code. Every decision you make must trace back to a requirement here.**

---

## 0. MISSION BRIEF

You are building a **production-grade datathon submission** with two parallel missions:

| Mission | Goal | Primary Output |
|---|---|---|
| **A — Predictive** | Predict whether a user will upgrade their plan within 7 days | Trained LightGBM model + FastAPI endpoint |
| **B — Architectural** | Classify users into behavioral funnel segments | Data-driven segmentation + KMeans validation + Streamlit dashboard |

Both missions share the same feature store. Build it once, use it twice.

> **REAL DATA NOTE:** The actual dataset columns are `person_id`, `timestamp`, and `event`
> (not `user_id`, `event_ts`, `event_type`). The upgrade event is `"subscription_upgraded"`.
> All column references go through `CFG` — update config once, it propagates everywhere.

---

## 1. ABSOLUTE CONSTRAINTS (never violate these)

### 1.1 Leakage Rules — HARDCODED, NON-NEGOTIABLE

The feature window for every user closes **24 hours before their earliest possible upgrade event**.
No feature may be computed using data after `user_cutoff_ts = upgrade_ts - INTERVAL '24 hours'`.

**Permanently banned column patterns** — if any of these strings appear in a feature name, DROP IT:
```
clicked_upgrade, viewed_pricing, started_trial, billing_event,
upgrade_initiated, conversion_flag, account_type, plan_type,
plan_name, is_premium, subscription_status, post_, _after_,
following_, next_, future_, redeem_upgrade
```

> ⚠️ The real data contains a `"redeem upgrade offer"` event. This is a **leakage signal**
> and must be banned. Using it will make the model look great in CV and fail in production.
> The judges are explicitly checking for this. Add it to BANNED_FEATURE_PATTERNS.

**Preprocessing leakage rule** — ALL preprocessing (scaling, encoding, imputation) must be
fitted ONLY on training data, NEVER on the full dataset. Use `sklearn.pipeline.Pipeline`
wrapping every transformer + the model. Pass the full pipeline into cross_val_score.

**Temporal aggregation rule** — every SQL/pandas aggregation must be anchored to
`user_cutoff_ts`. No `AVG(...) OVER (PARTITION BY person_id)` without a frame clause
bounding it to before the cutoff.

### 1.2 Code Quality Rules

- Every Python file must have a `if __name__ == "__main__":` block that can be run standalone
- Every function must have a docstring with Args and Returns
- Every file imports only from the standard library, installed packages, or sibling modules
- No Jupyter notebooks. Pure `.py` files only (Zerve runs Python scripts)
- All paths go through `CFG` (the config dataclass) — no hardcoded strings anywhere else
- Use `loguru` for all logging, not `print()` — except in the CLI entrypoint `main.py`
- All DataFrames passed between modules must be validated with `pandera` schemas

### 1.3 Dependency Rule

Install order matters. The build must work with:
```
python >= 3.10
lightgbm >= 4.0
scikit-learn >= 1.3
pandas >= 2.0
numpy >= 1.24
shap >= 0.43
fastapi >= 0.110
uvicorn >= 0.27
pydantic >= 2.0
streamlit >= 1.32
pandera >= 0.18
loguru >= 0.7
joblib >= 1.3
pyarrow >= 14.0   # for parquet I/O
matplotlib >= 3.8
seaborn >= 0.13
plotly >= 5.18
optuna >= 3.5     # hyperparameter tuning
duckdb >= 0.10    # SQL engine for parquet queries
requests >= 2.31  # for Streamlit → FastAPI calls
```

### 1.4 Zerve Compute Rules

> ⚠️ The dataset is 3M+ rows. Running it all at once on Lambda will hit the **15-minute AWS
> compute limit**. Follow these rules:

- **Use Fargate compute** (not Lambda) for: feature store construction, model training, any
  full-dataset SQL query. Fargate has no time limit.
- **Use Lambda** only for: EDA on samples, quick schema checks, API inference.
- **Sampling strategy for EDA:** Before building the full feature store, run a 10% stratified
  sample (stratified on whether user ever upgraded) to validate your feature logic. Once
  validated, run the full pipeline on Fargate.
- Add a `SAMPLE_FRAC` constant to `CFG` (default `1.0`; set to `0.1` for EDA runs).

---

## 2. REPOSITORY STRUCTURE

Build **exactly** this structure. Do not add extra files. Do not skip files.

```
datathon/
│
├── CLAUDE.md                    ← this file (already exists)
├── README.md                    ← setup + one-command run instructions
├── requirements.txt             ← all dependencies, pinned
├── Makefile                     ← convenience commands
│
├── config/
│   └── config.py                ← CFG dataclass (single source of truth)
│
├── data/
│   ├── raw/                     ← raw event CSVs/parquet go here (gitignored)
│   ├── processed/               ← feature_store.parquet, labels.parquet
│   └── .gitkeep
│
├── models/
│   ├── .gitkeep
│   └── (artifacts land here at runtime)
│
├── sql/
│   ├── 00_ingest_profile.sql    ← data quality audit queries
│   ├── 01_feature_store.sql     ← user-level feature aggregations (DuckDB compatible)
│   ├── 02_label_generation.sql  ← upgrade label with leakage buffer
│   └── 03_holdout_split.sql     ← time-based train/val/test split
│
├── src/
│   ├── __init__.py
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py            ← load raw events, validate schema
│   │   ├── schemas.py           ← pandera schemas for every DataFrame
│   │   └── sql_runner.py        ← DuckDB wrapper to run sql/ files
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   ├── leakage_audit.py     ← three-layer leakage sentinel
│   │   ├── engagement.py        ← feature group 1: volume metrics
│   │   ├── velocity.py          ← feature group 2: WoW growth
│   │   ├── recency.py           ← feature group 3: recency signals
│   │   ├── depth.py             ← feature group 4: feature breadth / Aha moment
│   │   ├── friction.py          ← feature group 5: error + rage-click signals
│   │   ├── social.py            ← feature group 6: collaboration signals
│   │   └── feature_store.py     ← orchestrator: joins all groups → feature_store.parquet
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mission_a_train.py   ← LightGBM + stratified CV + SHAP + artifact save
│   │   ├── mission_a_predict.py ← load artifact + run inference on new users
│   │   ├── hyperparameter_tune.py ← Optuna study for LGB params
│   │   └── evaluation.py        ← metrics: AUC, PR-AUC, calibration, confusion matrix
│   │
│   ├── funnel/
│   │   ├── __init__.py
│   │   ├── segments.py          ← data-driven segmentation (stages defined from EDA)
│   │   ├── clustering.py        ← KMeans validation + silhouette + PCA plot
│   │   ├── transitions.py       ← week-over-week segment transition matrix
│   │   └── funnel_report.py     ← segment distribution + upgrade rate per segment
│   │
│   └── api/
│       ├── __init__.py
│       ├── app.py               ← FastAPI application factory
│       ├── routes.py            ← /predict, /health, /model-info, /segments endpoints
│       ├── schemas.py           ← Pydantic v2 request/response models
│       └── middleware.py        ← CORS, request timing, error handling
│
├── dashboard/
│   └── app.py                   ← Streamlit dashboard (Mission B visualization)
│
├── tests/
│   ├── __init__.py
│   ├── test_leakage.py          ← assert banned features are caught
│   ├── test_features.py         ← assert feature shapes + no NaN in outputs
│   ├── test_api.py              ← assert API endpoints return correct schemas
│   └── test_segments.py         ← assert every user lands in exactly one segment
│
├── notebooks/
│   └── 00_eda_scratch.py        ← exploratory script (not part of pipeline)
│
└── main.py                      ← CLI entrypoint (runs full pipeline end to end)
```

---

## 3. BUILD ORDER

**Build files in this exact sequence.** Later files import earlier ones.
Never build a file before its dependencies exist.

```
Step 01  →  requirements.txt
Step 02  →  config/config.py
Step 03  →  src/data/schemas.py
Step 04  →  src/data/loader.py
Step 05  →  src/data/sql_runner.py
Step 06  →  sql/00_ingest_profile.sql
Step 07  →  sql/01_feature_store.sql
Step 08  →  sql/02_label_generation.sql
Step 09  →  sql/03_holdout_split.sql
Step 10  →  src/features/leakage_audit.py
Step 11  →  src/features/engagement.py
Step 12  →  src/features/velocity.py
Step 13  →  src/features/recency.py
Step 14  →  src/features/depth.py
Step 15  →  src/features/friction.py
Step 16  →  src/features/social.py
Step 17  →  src/features/feature_store.py
Step 18  →  src/models/evaluation.py
Step 19  →  src/models/hyperparameter_tune.py
Step 20  →  src/models/mission_a_train.py
Step 21  →  src/models/mission_a_predict.py
Step 22  →  src/funnel/segments.py
Step 23  →  src/funnel/clustering.py
Step 24  →  src/funnel/transitions.py
Step 25  →  src/funnel/funnel_report.py
Step 26  →  src/api/schemas.py
Step 27  →  src/api/middleware.py
Step 28  →  src/api/routes.py
Step 29  →  src/api/app.py
Step 30  →  dashboard/app.py
Step 31  →  tests/test_leakage.py
Step 32  →  tests/test_features.py
Step 33  →  tests/test_api.py
Step 34  →  tests/test_segments.py
Step 35  →  main.py
Step 36  →  Makefile
Step 37  →  README.md
Step 38  →  zerve_report.py       ← NEW: generate the required Zerve submission report
```

---

## 4. FILE-BY-FILE SPECIFICATIONS

### 4.1 `config/config.py`

A frozen `dataclasses.dataclass` called `DatathonConfig`. Instantiate a singleton `CFG` at
module level. Every other file does `from config.config import CFG`.

**Required fields:**
```python
# Paths
RAW_EVENTS_PATH: str = "data/raw/events.parquet"
FEATURE_STORE_PATH: str = "data/processed/feature_store.parquet"
LABELS_PATH: str = "data/processed/labels.parquet"
TRAIN_PATH: str = "data/processed/train.parquet"
VAL_PATH: str = "data/processed/val.parquet"
TEST_PATH: str = "data/processed/test.parquet"
MODEL_DIR: str = "models/"
SQL_DIR: str = "sql/"

# Temporal
FEATURE_WINDOW_DAYS: int = 30
LEAKAGE_BUFFER_HOURS: int = 24
PREDICTION_HORIZON_DAYS: int = 7
TRAIN_END_DATE: str = ""        # set at runtime from data
VAL_END_DATE: str = ""

# Model
RANDOM_STATE: int = 42
CV_FOLDS: int = 5
TARGET_COL: str = "will_upgrade_in_7d"

# ── REAL DATA COLUMN NAMES (updated from actual dataset) ──────────────────
USER_ID_COL: str = "person_id"          # was "user_id" — real col is person_id
TIMESTAMP_COL: str = "timestamp"         # was "event_ts"
EVENT_TYPE_COL: str = "event"            # was "event_type"
UPGRADE_EVENT_NAME: str = "subscription_upgraded"  # real upgrade event name

# Leakage — includes redeem_upgrade which is a known leakage event in this dataset
BANNED_FEATURE_PATTERNS: List[str] = [
    "clicked_upgrade", "viewed_pricing", "started_trial",
    "billing_event", "upgrade_initiated", "conversion_flag",
    "account_type", "plan_type", "plan_name", "is_premium",
    "subscription_status", "post_", "_after_", "following_",
    "next_", "future_", "redeem_upgrade",
]

# Zerve compute — set SAMPLE_FRAC to 0.1 for EDA/validation, 1.0 for full run
SAMPLE_FRAC: float = 1.0  # override to 0.1 for fast EDA on Lambda

# Funnel — DO NOT hardcode segment names here.
# Segments are defined in src/funnel/segments.py after EDA reveals actual behavior.
# This list is populated at runtime after segment discovery.
SEGMENT_NAMES: List[str] = dataclasses.field(default_factory=list)

# API
API_HOST: str = "0.0.0.0"
API_PORT: int = 8000

# Optuna
N_TRIALS: int = 50
```

Use `@dataclasses.dataclass(frozen=True)` so no field can be mutated at runtime.

---

### 4.2 `src/data/schemas.py`

Define `pandera.DataFrameSchema` objects for every major DataFrame shape.
Required schemas:

- `RawEventsSchema` — validates raw event log input
  - `person_id`: string, non-null  ← real column name
  - `timestamp`: datetime, non-null, no future dates  ← real column name
  - `event`: string, non-null  ← real column name
  - `session_id`: string, nullable (may not exist — make optional)

- `FeatureStoreSchema` — validates output of feature_store.py
  - `person_id`: string, non-null, unique  ← real column name
  - All numeric feature columns: float64 or int64, no inf values
  - No column names matching BANNED_FEATURE_PATTERNS

- `LabelSchema`
  - `person_id`: string, non-null, unique  ← real column name
  - `will_upgrade_in_7d`: int, values in {0, 1}
  - `user_cutoff_ts`: datetime, non-null

- `PredictionInputSchema` — matches Pydantic API schema for validation in tests

Each schema must have a `validate()` wrapper function that logs a warning (not raise)
for soft violations and raises `pandera.errors.SchemaError` for hard violations.

---

### 4.3 `src/data/loader.py`

```python
def load_raw_events(path: str = CFG.RAW_EVENTS_PATH, sample_frac: float = CFG.SAMPLE_FRAC) -> pd.DataFrame:
    """
    Load raw event data. Handles parquet and CSV.
    If sample_frac < 1.0, stratify sample on whether user has an upgrade event.
    Renames columns to internal names if needed (person_id, timestamp, event).
    Validates against RawEventsSchema.
    Casts timestamp to UTC-aware datetime.
    Returns validated DataFrame.
    """

def load_feature_store() -> pd.DataFrame:
    """Load and validate feature_store.parquet."""

def load_labels() -> pd.DataFrame:
    """Load and validate labels.parquet."""

def load_train_val_test() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load all three splits. Return (train, val, test)."""
```

Add a `__main__` block that loads raw events and prints shape + schema validation report.

---

### 4.4 `src/data/sql_runner.py`

Use **DuckDB** (not SQLite, not a DB server) to run the SQL files. DuckDB reads parquet
natively — no database setup needed.

```python
def run_sql_file(sql_path: str, output_path: str = None) -> pd.DataFrame:
    """
    Execute a .sql file using DuckDB.
    If output_path is given, save result as parquet.
    Returns result as DataFrame.
    DuckDB can directly query parquet files with:
        SELECT * FROM read_parquet('data/raw/events.parquet')
    Use this pattern inside the SQL files.
    """

def run_sql_string(sql: str) -> pd.DataFrame:
    """Execute a raw SQL string. For testing and EDA."""
```

---

### 4.5 SQL Files

All SQL files must be **DuckDB-compatible**. DuckDB syntax is mostly PostgreSQL-compatible
with these additions available: `read_parquet()`, `strptime()`, `epoch()`.

> **Column names in all SQL:** use `person_id`, `timestamp`, `event` — the real column names.

#### `sql/00_ingest_profile.sql`
Data quality report. Output: single-row summary per event type.
```sql
-- Queries to include:
-- 1. Total row count
-- 2. Date range (min/max timestamp)
-- 3. Distinct person_id count
-- 4. Event type distribution (event, count, pct) — top 30
-- 5. Null rates per column
-- 6. Users with 'subscription_upgraded' events (count + pct of total users)
-- 7. Median events per user
-- 8. P95 events per user
-- 9. Check for 'redeem upgrade offer' events — count + flag as LEAKAGE RISK
```

#### `sql/01_feature_store.sql`
User-level aggregations. **Every aggregation must be filtered to `timestamp <= user_cutoff_ts`**.
Join with the label table to get `user_cutoff_ts` per user.

Feature groups to compute in SQL:
```sql
-- Group 1: Engagement
total_events, unique_event_types, total_sessions, avg_events_per_session, active_days

-- Group 2: Velocity
events_last_7d, events_last_14d, events_last_30d,
wow_growth = (events_last_7d - events_last_14_to_7d) / (events_last_14_to_7d + 1.0),
is_accelerating = CASE WHEN wow_growth > 0.20 THEN 1 ELSE 0 END

-- Group 3: Recency
days_since_last_event, days_since_first_event, account_age_days

-- Group 4: Depth (adapt event names after running 00_ingest_profile.sql)
feature_breadth = COUNT(DISTINCT event) WHERE event IN (...core feature events...),
hit_aha_moment  = MAX(CASE WHEN event IN ('collaboration_event','integration_connected') THEN 1 ELSE 0 END)

-- Group 5: Friction
friction_events = COUNT(*) WHERE event IN ('$exception', 'error_event', 'failed_action'),
friction_ratio  = friction_events / NULLIF(total_events, 0)

-- Group 6: Social
social_actions  = COUNT(*) WHERE event IN ('invited_user','shared_content','commented'),
is_social_user  = CASE WHEN social_actions > 0 THEN 1 ELSE 0 END
```

> **After running 00_ingest_profile.sql**, review the actual event names in the data and
> update the `IN (...)` lists in Groups 4–6. Store these lists as constants in `config.py`.

#### `sql/02_label_generation.sql`
```sql
-- For each user who has a 'subscription_upgraded' event:
--   user_cutoff_ts = MIN(upgrade_ts) - INTERVAL '24 hours'
--   will_upgrade_in_7d = 1

-- For each user who does NOT have a 'subscription_upgraded' event:
--   user_cutoff_ts = MAX(timestamp) for that user
--   will_upgrade_in_7d = 0

-- A user is a negative example only if they were active (>=5 events) but never upgraded
-- This prevents inactive users from polluting the negative class
-- EXCLUDE any user whose events include 'redeem upgrade offer' from the negative class
-- (those users are likely mid-upgrade and the label is ambiguous)
```

#### `sql/03_holdout_split.sql`
```sql
-- Time-based split — NO random shuffling
-- Train: user_cutoff_ts <= P70 of all cutoff dates
-- Val:   user_cutoff_ts in (P70, P85]
-- Test:  user_cutoff_ts > P85
-- Save person_id + split assignment as processed/splits.parquet
-- This prevents temporal leakage at the split boundary
```

---

### 4.6 Feature Group Files (`src/features/`)

Each feature group file must implement a single public function:

```python
def compute_{group}(events_df: pd.DataFrame, cutoff_map: pd.Series) -> pd.DataFrame:
    """
    Args:
        events_df:  Raw events DataFrame (validated RawEventsSchema)
        cutoff_map: pd.Series with index=person_id, values=user_cutoff_ts
                    Use this to filter events to before the cutoff for each user.
    Returns:
        pd.DataFrame indexed by person_id with computed features.
        All column names must be snake_case, no banned patterns.
    """
```

The `cutoff_map` pattern is critical. Implement it like this:
```python
# Filter events to before each user's cutoff timestamp
df = events_df.copy()
df = df.merge(cutoff_map.rename('cutoff_ts'), on=CFG.USER_ID_COL, how='inner')
df = df[df[CFG.TIMESTAMP_COL] <= df['cutoff_ts']]
# Now all aggregations are safe
```

**`engagement.py`** — total_events, unique_event_types, total_sessions,
avg_events_per_session, active_days

**`velocity.py`** — events_last_7d, events_last_14d, events_last_30d,
wow_growth, is_accelerating, events_last_14_to_7d

**`recency.py`** — days_since_last_event, days_since_first_event, account_age_days

**`depth.py`** — feature_breadth (count of distinct core event types used),
hit_aha_moment (bool: did user use a collaboration or integration event)
Core events list: define as a module-level constant, easy to customize.

**`friction.py`** — friction_events (count), friction_ratio (fraction of total events
that are error/timeout/failed_action/rage_click type). Include `$exception` events.

**`social.py`** — social_actions (count), is_social_user (bool)

---

### 4.7 `src/features/leakage_audit.py`

Three-layer leakage sentinel. Must be called before ANY model training.

```python
@dataclasses.dataclass
class LeakageReport:
    passed: bool
    banned_found: List[str]          # Layer 1: name pattern matches
    temporal_violations: List[str]   # Layer 2: temporal keyword scan
    high_correlation_suspects: List[str]  # Layer 3: r > 0.85 with target
    details: Dict[str, Any]

def audit_features(
    df: pd.DataFrame,
    target_col: str = CFG.TARGET_COL,
    correlation_threshold: float = 0.85,
) -> LeakageReport:
    """Three-layer leakage defense. Returns full report."""

def enforce_audit(df: pd.DataFrame, hard_stop: bool = True) -> pd.DataFrame:
    """
    Run audit. Auto-drop banned columns. Raise on unresolved temporal violations
    if hard_stop=True. Log all actions via loguru.
    Returns cleaned DataFrame.
    """
```

Layer 3 implementation detail: sample 10,000 rows max for correlation check to keep it fast.
Log a WARNING (not error) for correlation suspects — they may be legitimate features.

---

### 4.8 `src/features/feature_store.py`

Orchestrator. Calls all six compute functions, joins results, validates with
`FeatureStoreSchema`, runs `enforce_audit()`, saves to parquet.

```python
def build_feature_store(
    events_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    save: bool = True,
) -> pd.DataFrame:
    """
    1. Build cutoff_map from labels_df (person_id → user_cutoff_ts)
    2. Call all six compute_* functions with the same cutoff_map
    3. Left-join all feature groups on person_id (start with labels_df users as spine)
    4. Fill NaN with 0 (users who never triggered a group's events)
    5. Run enforce_audit() — raises if hard violations found
    6. Validate with FeatureStoreSchema
    7. If save=True, write to CFG.FEATURE_STORE_PATH
    8. Log feature group row counts and final shape
    Returns feature_store DataFrame
    """
```

---

### 4.9 `src/models/evaluation.py`

```python
def full_eval_report(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.5,
    dataset_name: str = "validation",
) -> Dict[str, float]:
    """
    Compute and log:
    - ROC-AUC
    - PR-AUC (average_precision_score) — primary metric for imbalanced data
    - F1 at threshold
    - Precision, Recall at threshold
    - Brier score (calibration)
    - Log loss
    Returns dict of all metrics.
    """

def plot_calibration_curve(y_true, y_pred_proba, save_path: str): ...
def plot_roc_curve(y_true, y_pred_proba, save_path: str): ...
def plot_pr_curve(y_true, y_pred_proba, save_path: str): ...
def plot_confusion_matrix(y_true, y_pred, save_path: str): ...

def threshold_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> pd.DataFrame:
    """
    For thresholds 0.1 to 0.9 in 0.05 steps:
    compute precision, recall, F1, and % of users flagged.
    Return as DataFrame. Helps judges pick operational threshold.
    """
```

---

### 4.10 `src/models/hyperparameter_tune.py`

Use **Optuna** with a `LightGBMTuner` objective. Time-box to 30 minutes maximum.

```python
def run_optuna_study(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = CFG.N_TRIALS,
    timeout_seconds: int = 1800,  # 30 min hard cap
) -> Dict[str, Any]:
    """
    Optuna study optimizing PR-AUC (not AUC) via 3-fold CV.
    Search space:
        num_leaves: 20-300
        learning_rate: 1e-3 to 0.3 (log scale)
        min_child_samples: 10-100
        feature_fraction: 0.5-1.0
        bagging_fraction: 0.5-1.0
        lambda_l1: 1e-8 to 10 (log scale)
        lambda_l2: 1e-8 to 10 (log scale)
        max_depth: 3-12
    Returns best params dict.
    """
```

---

### 4.11 `src/models/mission_a_train.py`

The main training orchestrator. Must be runnable standalone.

```python
def train(
    use_tuned_params: bool = False,
    save_artifacts: bool = True,
) -> Tuple[lgb.LGBMClassifier, Dict[str, Any]]:
    """
    Full training pipeline:

    1. Load feature store + labels
    2. enforce_audit() — hard stop on violations
    3. Merge features + labels on person_id
    4. Time-based train/val/test split (from splits.parquet)
    5. Build sklearn Pipeline:
       a. SimpleImputer(strategy='median')
       b. (No scaler — LightGBM doesn't need scaling)
       c. LGBMClassifier with is_unbalance=True
    6. If use_tuned_params=True: load best params from hyperparameter_tune.py
       Else: use sensible defaults (num_leaves=63, lr=0.05, n_estimators=1000)
    7. StratifiedKFold(n_splits=CFG.CV_FOLDS) cross-validation
       - Track OOF predictions for each fold
       - Use early_stopping(50) on each fold's validation set
    8. Compute OOF AUC + OOF PR-AUC — log both prominently
    9. Evaluate best fold model on held-out TEST set
    10. Run SHAP TreeExplainer on 1000 test samples
        - Save shap_values.npy
        - Save shap_summary.png (beeswarm plot, top 20 features)
    11. Save artifacts:
        - models/upgrade_predictor.pkl   (best fold pipeline)
        - models/feature_names.pkl
        - models/model_metadata.json
        - models/oof_predictions.csv
        - models/fold_scores.csv
        - models/shap_summary.png
        - models/evaluation/  (all eval plots)
    12. Print final summary table to stdout

    Returns (best_pipeline, metadata_dict)
    """
```

**SHAP requirement:** The SHAP beeswarm plot must use `shap.plots.beeswarm()` with
`max_display=20`. Save at 200dpi. This is a judging artifact.

---

### 4.12 `src/models/mission_a_predict.py`

```python
def load_model() -> Tuple[Pipeline, List[str], Dict]:
    """Load pipeline, feature names, and metadata from models/."""

def predict_single(user_features: Dict[str, float]) -> Dict[str, Any]:
    """
    Run inference for one user.
    Returns:
    {
        "upgrade_probability": float,
        "upgrade_risk_tier": str,   # HOT/WARM/NURTURE/COLD
        "recommended_action": str,
        "model_version": str,
        "feature_count": int,
    }
    """

def predict_batch(feature_store: pd.DataFrame) -> pd.DataFrame:
    """
    Run inference on entire feature store.
    Add columns: upgrade_probability, upgrade_risk_tier, recommended_action.
    Return augmented DataFrame.
    """

def get_risk_tier(prob: float) -> Tuple[str, str]:
    """
    HOT_LEAD    → prob >= 0.75 → "Trigger in-app upgrade modal"
    WARM_LEAD   → prob >= 0.50 → "Send personalized feature highlight email"
    NURTURE     → prob >= 0.25 → "Enroll in drip campaign"
    COLD        → prob  < 0.25 → "Focus on Aha moment activation"
    """
```

---

### 4.13 `src/funnel/segments.py`

> **IMPORTANT CHANGE FROM ORIGINAL SPEC:** Do NOT hardcode 4 segment names before seeing
> the data. The judges are evaluating whether your funnel is grounded in real behavior.
> Follow this two-phase approach:
>
> **Phase 1 (EDA — run before writing this file):**
> After running `00_ingest_profile.sql` and exploring the event distribution, define your
> funnel stages based on what you actually see. Look at the example 8-stage funnel in the
> brief (New → Exploring → Created Content → Used AI → Wrote Code → Integrated → Engaged
> → Upgraded) and adapt it to the real event names in this dataset.
>
> **Phase 2 (Implementation):**
> Each stage must have a clear, deterministic rule based on observable events and thresholds.
> A judge should be able to read the rule and implement it themselves.

```python
# Define SEGMENT_DEFINITIONS after EDA. Structure:
SEGMENT_DEFINITIONS = {
    "StageName": {
        "description": "What this user has done / where they are",
        "rules": "Explicit boolean rule — events, counts, thresholds, time windows",
        "action": "What the product team should do for this user",
        "upgrade_likelihood": "Low / Medium / High",
    },
    # ... more stages
}
# Rules for good segments (from judging rubric):
# - Specific & Observable: each stage has a clear definition based on real events
# - Deterministic Transitions: no ambiguity, no subjective calls
# - Complete Coverage: every user fits somewhere, no uncategorized users
# - Time-Aware: stages can change over time based on events + time windows

def assign_segments(feature_store: pd.DataFrame) -> pd.DataFrame:
    """
    Apply segment rules via np.select.
    Adds 'funnel_segment' column.
    Unclassified users get segment 'Unclassified' (should be < 2% if rules are correct).
    Logs segment distribution.
    Raises if > 10% of users are Unclassified (means rules need tuning).
    """

def get_segment_stats(df_with_segments: pd.DataFrame) -> pd.DataFrame:
    """
    For each segment, compute:
    - count, pct_of_total
    - avg_upgrade_probability (if upgrade_probability column exists)
    - avg_active_days, avg_feature_breadth, avg_friction_ratio
    Return as summary DataFrame.
    """
```

---

### 4.14 `src/funnel/clustering.py`

```python
CLUSTER_FEATURES = [
    "total_events", "unique_event_types", "total_sessions",
    "wow_growth", "days_since_last_event", "feature_breadth",
    "hit_aha_moment", "friction_ratio", "social_actions",
    "active_days", "is_accelerating",
]

def run_clustering(
    feature_store: pd.DataFrame,
    n_clusters: int = 4,
) -> Tuple[pd.DataFrame, KMeans, RobustScaler]:
    """
    1. Select CLUSTER_FEATURES (warn + drop any not present)
    2. RobustScaler().fit_transform()
    3. Silhouette analysis: k=2..8, find best_k
    4. Fit KMeans(n_clusters, random_state=CFG.RANDOM_STATE, n_init=20)
    5. Add 'cluster_id' column to feature_store
    6. PCA(2 components) for visualization
    7. Save cluster_visualization.png — two panels:
       Left: KMeans cluster coloring
       Right: Rule-based segment coloring
       (This comparison validates that rules align with data geometry)
    8. Save kmeans.pkl and cluster_scaler.pkl to models/
    Returns (df_with_clusters, fitted_kmeans, fitted_scaler)
    """

def cluster_profile(df_with_clusters: pd.DataFrame) -> pd.DataFrame:
    """
    For each cluster_id, compute mean of all CLUSTER_FEATURES.
    Transpose so features are rows, clusters are columns.
    This is the cluster interpretation table for the presentation.
    """
```

---

### 4.15 `src/funnel/transitions.py`

```python
def compute_transition_matrix(
    events_df: pd.DataFrame,
    feature_store: pd.DataFrame,
    window_weeks: int = 4,
) -> pd.DataFrame:
    """
    For each pair of consecutive weeks in the data:
    1. Re-compute segment assignment using only events up to end of week N
    2. Re-compute segment assignment using only events up to end of week N+1
    3. Build a transition matrix: P(segment_t+1 | segment_t)

    This answers: "What % of users in stage X move to stage Y each week?"

    Returns: normalized transition probability matrix as DataFrame
             (rows = from_segment, cols = to_segment)
    """

def plot_transition_heatmap(transition_matrix: pd.DataFrame, save_path: str):
    """
    Seaborn heatmap. Annotate each cell with the probability.
    Title: "Weekly Segment Transition Probabilities"
    This is a key judging artifact (Rubric item 5: Transition Logic).
    """
```

---

### 4.16 `src/funnel/funnel_report.py`

```python
def generate_full_report(
    feature_store: pd.DataFrame,
    df_with_segments: pd.DataFrame,
    df_with_predictions: pd.DataFrame = None,
) -> Dict[str, Any]:
    """
    Combine Mission A predictions with Mission B segments.
    For each segment, compute:
    - User count + %
    - Mean upgrade_probability (from Mission A, if available)
    - Top 3 most diagnostic features (highest mean |SHAP| within segment)
    - Recommended business action
    - Conversion rate from this stage (% who eventually upgraded)

    Save full report to models/funnel_report.json
    Save segment_upgrade_proba.png — bar chart of mean upgrade prob per segment
    Return report dict.
    """
```

---

### 4.17 API Files (`src/api/`)

#### `schemas.py` — Pydantic v2 models

```python
class UserFeatureInput(BaseModel):
    """All features the model needs. All numeric. No banned columns."""
    model_config = ConfigDict(extra="forbid")  # reject unknown fields

    total_events: int = Field(..., ge=0, description="Total events in window")
    unique_event_types: int = Field(..., ge=0)
    total_sessions: int = Field(..., ge=0)
    avg_events_per_session: float = Field(..., ge=0.0)
    active_days: int = Field(..., ge=0)
    wow_growth: float = Field(default=0.0)
    is_accelerating: Literal[0, 1] = 0
    events_last_7d: int = Field(default=0, ge=0)
    events_last_14d: int = Field(default=0, ge=0)
    events_last_30d: int = Field(default=0, ge=0)
    days_since_last_event: int = Field(..., ge=0)
    days_since_first_event: int = Field(..., ge=0)
    account_age_days: int = Field(..., ge=0)
    feature_breadth: int = Field(default=0, ge=0)
    hit_aha_moment: Literal[0, 1] = 0
    friction_events: int = Field(default=0, ge=0)
    friction_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    social_actions: int = Field(default=0, ge=0)
    is_social_user: Literal[0, 1] = 0

class PredictionResponse(BaseModel):
    person_id: Optional[str] = None   # real ID field name
    upgrade_probability: float
    upgrade_risk_tier: str
    funnel_segment: str
    cluster_id: int
    top_risk_factors: Dict[str, str]
    recommended_action: str
    model_version: str
    inference_time_ms: float

class BatchPredictionRequest(BaseModel):
    users: List[UserFeatureInput]
    person_ids: Optional[List[str]] = None   # real ID field name

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    oof_auc: float
    oof_pr_auc: float
    training_samples: int
    feature_count: int
    api_version: str = "1.0.0"
```

#### `middleware.py`

```python
# Implement:
# 1. CORSMiddleware (allow all origins for datathon)
# 2. Request timing middleware — add X-Process-Time header to every response
# 3. Global exception handler — return {"error": str(e), "type": type(e).__name__}
#    for all unhandled exceptions (never expose stack traces)
# 4. Request ID middleware — add X-Request-ID UUID header
```

#### `routes.py`

Implement these endpoints:

```
GET  /health
     → HealthResponse: model loaded status + OOF metrics

POST /predict
     → Body: UserFeatureInput
     → Response: PredictionResponse
     → Include inference timing in response

POST /predict/batch
     → Body: BatchPredictionRequest (up to 1000 users)
     → Response: List[PredictionResponse]
     → Process in vectorized batch, not a loop

GET  /model-info
     → Full model_metadata.json contents

GET  /segments
     → Segment definitions from SEGMENT_DEFINITIONS dict
     → Include count + pct if feature store is loaded

GET  /segments/stats
     → get_segment_stats() output as JSON

GET  /docs
     → FastAPI auto-generates this — just make sure it works
```

All prediction endpoints must:
1. Validate input with Pydantic (automatic)
2. Reject requests with any banned feature names in extra fields
3. Return HTTP 503 if model is not loaded (not 500)
4. Log every request with loguru at INFO level

#### `app.py`

```python
def create_app() -> FastAPI:
    """
    Application factory pattern.
    1. Create FastAPI(title="User Intelligence API", version="1.0.0")
    2. Add all middleware from middleware.py
    3. Include router from routes.py
    4. Add startup event: load all model artifacts
    5. Add shutdown event: log graceful shutdown
    6. Return app
    """

app = create_app()  # module-level for uvicorn

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host=CFG.API_HOST, port=CFG.API_PORT, reload=False)
```

---

### 4.18 `dashboard/app.py` — Streamlit Dashboard

Use Streamlit. The dashboard must have these **four tabs**:

**Tab 1: Overview**
- 4 metric cards: total users, top segment %, upgrade rate %, model PR-AUC
- Funnel bar chart: segment counts (plotly bar, horizontal)
- Upgrade probability distribution histogram by segment

**Tab 2: Segment Explorer**
- Dropdown: select a segment
- Show: user count, avg upgrade probability, top features for this segment
- Show: segment definition + recommended business action
- Show: transition probabilities out of this segment (where do users go next?)

**Tab 3: Upgrade Predictor (Live)**
- Input form: all UserFeatureInput fields as sliders/number inputs
- "Predict" button → calls the FastAPI /predict endpoint
- Display: probability gauge, risk tier badge, funnel segment, recommended action
- If API is not running, show a clear "API offline — run uvicorn first" warning
- API URL configurable via `st.sidebar` text input (default: http://localhost:8000)

**Tab 4: Model Performance**
- Load and display: fold_scores.csv as a table
- Display: shap_summary.png if it exists
- Display: OOF AUC and PR-AUC as big metric cards
- Display: threshold_analysis table from evaluation.py

```python
# Streamlit config at top of file:
st.set_page_config(
    page_title="User Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

---

### 4.19 Tests (`tests/`)

#### `test_leakage.py`
```python
def test_banned_features_detected():
    """Create a DataFrame with banned column names. Assert audit fails."""

def test_redeem_upgrade_is_banned():
    """Assert that a feature named 'redeem_upgrade_count' is caught by the audit.
    This is the specific leakage signal called out in the hackathon brief."""

def test_temporal_keywords_detected():
    """Create columns named 'events_after_signup', 'post_upgrade_sessions'. Assert detected."""

def test_clean_features_pass():
    """Create valid feature set. Assert audit passes."""

def test_enforce_audit_drops_banned():
    """Assert banned columns are dropped and clean columns are preserved."""
```

#### `test_features.py`
```python
def test_engagement_no_nulls(sample_events):
    """Assert compute_engagement returns no NaN values."""

def test_velocity_shape(sample_events):
    """Assert output has one row per unique person_id."""

def test_cutoff_map_enforced(sample_events):
    """
    Insert a future event (after cutoff_ts) for one user.
    Assert it does NOT affect that user's feature values.
    This is the most important test in the codebase.
    """

def test_feature_store_schema_valid(sample_feature_store):
    """Assert FeatureStoreSchema.validate() passes on real output."""
```

#### `test_segments.py`
```python
def test_every_user_has_exactly_one_segment(sample_features):
    """Assert no person_id appears in multiple segments or zero segments."""

def test_segment_names_match_config(sample_features):
    """Assert all segment values are in CFG.SEGMENT_NAMES + ['Unclassified']."""

def test_unclassified_rate_below_threshold(real_feature_store):
    """Assert < 10% of users are Unclassified."""
```

#### `test_api.py`
Use `fastapi.testclient.TestClient`.
```python
def test_health_returns_200(): ...
def test_predict_valid_input_returns_200(): ...
def test_predict_banned_field_returns_422(): ...
def test_predict_missing_required_field_returns_422(): ...
def test_predict_probability_between_0_and_1(): ...
def test_batch_predict_returns_correct_count(): ...
```

---

### 4.20 `main.py` — Full Pipeline CLI

```python
"""
CLI entrypoint for the full pipeline.

Usage:
    python main.py --help
    python main.py ingest              # run SQL profiling
    python main.py features            # build feature store
    python main.py train               # train Mission A model
    python main.py train --tune        # with Optuna hyperparameter search
    python main.py funnel              # run Mission B segmentation
    python main.py report              # generate combined funnel report
    python main.py api                 # launch FastAPI server
    python main.py dashboard           # launch Streamlit dashboard
    python main.py zerve-report        # generate Zerve submission report
    python main.py all                 # run full pipeline end to end
    python main.py test                # run pytest suite
"""

import argparse
# All steps as individual functions that call the relevant modules
# Full pipeline: ingest → features → train → funnel → report → zerve-report
# Clear success/failure logging at each step
# Total wall-clock time printed at end of 'all' command
```

---

### 4.21 `Makefile`

```makefile
.PHONY: install ingest features train funnel api dashboard report zerve-report test all clean

install:
	pip install -r requirements.txt

ingest:
	python main.py ingest

features:
	python main.py features

train:
	python main.py train

tune:
	python main.py train --tune

funnel:
	python main.py funnel

api:
	python main.py api

dashboard:
	python main.py dashboard

report:
	python main.py report

zerve-report:
	python main.py zerve-report

test:
	pytest tests/ -v --tb=short

all:
	python main.py all

clean:
	rm -rf data/processed/*.parquet models/*.pkl models/*.json models/*.png
```

---

### 4.22 `README.md`

Must include:
1. **One-command setup**: `pip install -r requirements.txt && make all`
2. Data format expected (column names, types) — `person_id`, `timestamp`, `event`
3. How to customize event names (where in config to change)
4. How to run the API and test it with curl
5. How to launch the dashboard
6. Key results section (fill in after training)
7. Architecture diagram (ASCII is fine)
8. Leakage prevention section — explain the three layers + why `redeem upgrade offer` is banned

---

### 4.23 `zerve_report.py` — Zerve Submission Report ← NEW

> This file generates the **required Zerve Report** submission artifact.
> The report must be generated using the Zerve agent inside the platform.
> This script prepares all the content for it.

```python
"""
Generates the Zerve submission report content.

The Zerve Report (required deliverable) must cover:
1. Approach summary — how you framed both missions
2. Feature engineering decisions — which features, why, leakage handling
3. Model design — LightGBM, CV strategy, why PR-AUC as primary metric
4. Funnel definition — stage names, explicit rules, transition logic
5. Business implications — what should the product team do with this?
6. Key results — OOF PR-AUC, segment distribution, upgrade rates by segment

This script loads all artifacts and assembles a structured markdown report.
It is then copy-pasted into the Zerve agent to generate the final Zerve Report.
"""

def generate_report_content() -> str:
    """
    Load: model_metadata.json, funnel_report.json, fold_scores.csv,
          cluster_visualization.png, shap_summary.png, transition_matrix
    Assemble into a structured markdown document.
    Print to stdout and save to models/zerve_report_draft.md
    """
```

---

## 5. DATA FORMAT

The actual dataset has these columns (confirmed from hackathon brief):

```
events.parquet columns:
    person_id     string     unique user identifier  (NOT user_id)
    timestamp     datetime   UTC timestamp of event  (NOT event_ts)
    event         string     name of the event       (NOT event_type)
                             includes "subscription_upgraded" as the upgrade event
                             includes "$exception" for errors
                             includes "redeem upgrade offer" — BANNED LEAKAGE SIGNAL
```

If additional columns exist (e.g., `properties`, `session_id`), add them to
`RawEventsSchema` as nullable and use them if useful. The ONLY place to update column
references is `config/config.py`.

---

## 6. CRITICAL IMPLEMENTATION NOTES

### On the LightGBM Pipeline

Do NOT use `LGBMClassifier` directly in cross_val_score.
Always wrap in a Pipeline:
```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("model", LGBMClassifier(**lgb_params)),
])
# Then pass `pipeline` to StratifiedKFold loop
# fit on X_train, predict on X_val — never the other way
```

### On SHAP with Pipelines

To use TreeExplainer on a pipeline:
```python
explainer = shap.TreeExplainer(pipeline.named_steps["model"])
# Transform X through all steps EXCEPT the final model
X_transformed = pipeline[:-1].transform(X_sample)
shap_values = explainer.shap_values(X_transformed)
```

### On DuckDB + Parquet

DuckDB can directly read parquet files without loading them into memory:
```sql
SELECT * FROM read_parquet('data/raw/events.parquet') LIMIT 10;
```
Use this pattern in all SQL files. Do not load entire datasets into pandas before SQL ops.

### On Zerve Compute (Lambda vs Fargate)

- **Lambda** (default): 15-min hard limit from AWS. Use only for quick tasks.
- **Fargate**: No time limit. Use for full feature store build and model training.
- Switch compute type in the Zerve canvas cell settings before running heavy steps.
- For the 3M+ row dataset, feature store construction will exceed Lambda limits.

### On Streamlit + FastAPI co-existence

Run them on different ports:
- FastAPI: port 8000 (default)
- Streamlit: port 8501 (default)
Both can run simultaneously in separate Zerve canvas cells.
The Streamlit app calls the FastAPI endpoint via `requests.post("http://localhost:8000/predict")`.

### On imbalanced classes

If positive rate < 5%:
- Use `is_unbalance=True` in LGBMClassifier
- Report PR-AUC as primary metric (not AUC) — the judges explicitly look for this
- Report threshold analysis so judges can see precision/recall tradeoffs
- Do NOT oversample (SMOTE etc.) — temporal data ordering matters

### On Funnel Design (Judging Rubric Item 4 & 5 — 50 pts combined)

The judges award 17-25 pts each for:
- Funnel design that is "clear, well-justified, grounded in real behavior"
- Transition logic that uses "explicit definitions, sequences, time windows"

This means your segment rules must reference actual event names from the data, not
abstract concepts. After running EDA, identify the real behavioral milestones (e.g.,
"first time user sent a message to the agent", "first time user connected an integration")
and build your funnel around those.

---

## 7. WHAT "DONE" LOOKS LIKE

The submission is complete when ALL of the following are true:

- [ ] `make install && make all` runs without errors on a clean environment
- [ ] `make test` passes all tests (zero failures)
- [ ] `make api` launches FastAPI; `curl http://localhost:8000/health` returns 200
- [ ] `make dashboard` launches Streamlit on port 8501
- [ ] `models/upgrade_predictor.pkl` exists and loads cleanly
- [ ] `models/shap_summary.png` exists and shows top 20 features
- [ ] `models/funnel_report.json` exists with all segments populated
- [ ] `models/zerve_report_draft.md` exists with full report content
- [ ] `models/cluster_visualization.png` exists (KMeans vs rule-based comparison)
- [ ] `models/transition_heatmap.png` exists
- [ ] OOF PR-AUC is logged to stdout during training
- [ ] Zero banned features present in the feature store (audit must pass)
- [ ] Every user in the feature store has exactly one funnel segment
- [ ] `redeem_upgrade` patterns are not present in any feature name (audit enforces this)

**Submission checklist (due 10:00 AM Thursday):**
- [ ] Zerve Project link (runnable, clean, documented)
- [ ] Zerve Report link (generated with Zerve agent from zerve_report_draft.md)
- [ ] 3-minute video (recorded portrait/vertical on phone)
- [ ] Optional: Deployment artifact link (FastAPI app or interactive Streamlit app)

---

## 8. AGENT BEHAVIOR RULES

1. **Build in order.** Follow the Step 01-38 sequence exactly. Do not skip steps.
2. **No placeholders.** Every file must be complete and runnable. No `# TODO` comments.
3. **Test as you go.** After building each module, add its `if __name__ == "__main__"` block
   and verify the imports resolve.
4. **Run EDA before defining funnel segments.** After Step 06 (ingest profile SQL), review
   the actual event names and frequencies. Use those to define segment rules in Step 22.
   Do not invent segment names before seeing the data.
5. **Leakage is sacred.** If you are unsure whether a feature is safe, implement the
   conservative version (exclude it) and log a WARNING explaining why. The `redeem upgrade
   offer` event is a confirmed leakage signal — never use it as a feature.
6. **Never hardcode paths or column names.** All paths go through `CFG`. All column
   references use `CFG.USER_ID_COL`, `CFG.TIMESTAMP_COL`, `CFG.EVENT_TYPE_COL`. No exceptions.
7. **Use Fargate for heavy compute.** Switch the canvas cell to Fargate before running
   feature store construction or model training on the full 3M+ row dataset.
8. **Report progress.** After building each file, output a one-line confirmation:
   `✓ Built: path/to/file.py — [brief description of what it does]`
