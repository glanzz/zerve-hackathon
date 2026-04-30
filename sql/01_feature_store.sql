-- 01_feature_store.sql
-- Build per-user feature vectors from data/raw/events.parquet.
-- All aggregations start from `clean_events`, which strips banned events
-- and enforces the timestamp <= user_cutoff_ts boundary.
--
-- Feature groups
--   core      : basic activity counters
--   depth     : agent / generation / tooling events
--   friction  : exception events
--   account   : account age
--   recency   : days since last event

WITH

  -- ── 0. Raw source ──────────────────────────────────────────────────────
  raw AS (
    SELECT * FROM read_parquet('data/raw/events.parquet')
  ),

  -- ── 1. Cutoff anchor: each user's upgrade ts (or MAX ts if no upgrade) ─
  user_cutoff AS (
    SELECT
      person_id,
      COALESCE(
        MIN(CASE WHEN event = 'subscription_upgraded' THEN timestamp END),
        MAX(timestamp)
      ) AS user_cutoff_ts
    FROM raw
    GROUP BY person_id
  ),

  -- ── 2. Clean events: banned list + cutoff filter ────────────────────────
  --      THIS IS THE ONLY PLACE WHERE RAW EVENTS ARE FILTERED.
  --      ALL feature CTEs reference clean_events, not raw.
  clean_events AS (
    SELECT r.*
    FROM raw r
    JOIN user_cutoff uc USING (person_id)
    WHERE r.event NOT IN (
        'subscription_upgraded',
        'promo_code_redeemed',
        'addon_credits_used',
        'offer_declined',
        'credits_exceeded',
        'credits_below_1',
        'credits_below_2',
        'credits_below_3',
        'credits_below_4',
        'notebook_deployment_usage_tracked'
      )
      AND r.timestamp <= uc.user_cutoff_ts
  ),

  -- ── 3. Core features ───────────────────────────────────────────────────
  feat_core AS (
    SELECT
      person_id,
      COUNT(*)                                         AS total_events,
      COUNT(DISTINCT DATE_TRUNC('day', timestamp))     AS active_days,
      COUNT(DISTINCT DATE_TRUNC('week', timestamp))    AS active_weeks,
      COUNT(DISTINCT event)                            AS distinct_event_types,
      MIN(timestamp)                                   AS first_event_ts,
      MAX(timestamp)                                   AS last_event_ts
    FROM clean_events
    GROUP BY person_id
  ),

  -- ── 4. Depth features (agent / AI / tooling events) ────────────────────
  feat_depth AS (
    SELECT
      person_id,
      COUNT(CASE WHEN event = 'agent_new_chat'                         THEN 1 END) AS depth_agent_new_chat_count,
      COUNT(CASE WHEN event = '$ai_generation'                         THEN 1 END) AS depth_ai_generation_count,
      COUNT(CASE WHEN event = 'run_block'                              THEN 1 END) AS depth_run_block_count,
      COUNT(CASE WHEN event = 'agent_tool_call_create_block_tool'      THEN 1 END) AS depth_tool_create_block_count,
      COUNT(CASE WHEN event = 'agent_tool_call_run_block_tool'         THEN 1 END) AS depth_tool_run_block_count,
      COUNT(CASE WHEN event = 'agent_tool_call_get_block_tool'         THEN 1 END) AS depth_tool_get_block_count,
      COUNT(CASE WHEN event IN (
              'agent_new_chat',
              '$ai_generation',
              'run_block',
              'agent_tool_call_create_block_tool',
              'agent_tool_call_run_block_tool',
              'agent_tool_call_get_block_tool'
            ) THEN 1 END)                                                          AS depth_total_ai_tool_events
    FROM clean_events
    GROUP BY person_id
  ),

  -- ── 5. Friction features ($exception only) ─────────────────────────────
  feat_friction AS (
    SELECT
      person_id,
      COUNT(CASE WHEN event = '$exception' THEN 1 END)  AS friction_exception_count
    FROM clean_events
    GROUP BY person_id
  ),

  -- ── 6. Account-age feature ─────────────────────────────────────────────
  --       Uses new_user_created (NOT sign_up) as the tenure anchor.
  feat_account AS (
    SELECT
      r.person_id,
      DATEDIFF('day',
        MIN(CASE WHEN r.event = 'new_user_created' THEN r.timestamp END),
        uc.user_cutoff_ts
      ) AS account_age_days
    FROM raw r
    JOIN user_cutoff uc USING (person_id)
    GROUP BY r.person_id, uc.user_cutoff_ts
  ),

  -- ── 7. Recency feature ─────────────────────────────────────────────────
  feat_recency AS (
    SELECT
      ce.person_id,
      DATEDIFF('day', MAX(ce.timestamp), uc.user_cutoff_ts) AS days_since_last_event
    FROM clean_events ce
    JOIN user_cutoff uc USING (person_id)
    GROUP BY ce.person_id, uc.user_cutoff_ts
  ),

  -- ── 8. All users from clean_events ─────────────────────────────────────
  all_users AS (
    SELECT DISTINCT person_id FROM clean_events
  )

-- ── 9. Final join ──────────────────────────────────────────────────────────
SELECT
  u.person_id,
  uc.user_cutoff_ts,

  -- core
  COALESCE(fc.total_events,        0) AS total_events,
  COALESCE(fc.active_days,         0) AS active_days,
  COALESCE(fc.active_weeks,        0) AS active_weeks,
  COALESCE(fc.distinct_event_types,0) AS distinct_event_types,
  fc.first_event_ts,
  fc.last_event_ts,

  -- depth
  COALESCE(fd.depth_agent_new_chat_count,    0) AS depth_agent_new_chat_count,
  COALESCE(fd.depth_ai_generation_count,     0) AS depth_ai_generation_count,
  COALESCE(fd.depth_run_block_count,         0) AS depth_run_block_count,
  COALESCE(fd.depth_tool_create_block_count, 0) AS depth_tool_create_block_count,
  COALESCE(fd.depth_tool_run_block_count,    0) AS depth_tool_run_block_count,
  COALESCE(fd.depth_tool_get_block_count,    0) AS depth_tool_get_block_count,
  COALESCE(fd.depth_total_ai_tool_events,    0) AS depth_total_ai_tool_events,

  -- friction
  COALESCE(ff.friction_exception_count, 0)      AS friction_exception_count,

  -- account
  fa.account_age_days,

  -- recency
  COALESCE(fr.days_since_last_event, 0)         AS days_since_last_event

FROM all_users u
JOIN user_cutoff uc USING (person_id)
LEFT JOIN feat_core    fc USING (person_id)
LEFT JOIN feat_depth   fd USING (person_id)
LEFT JOIN feat_friction ff USING (person_id)
LEFT JOIN feat_account  fa USING (person_id)
LEFT JOIN feat_recency  fr USING (person_id)
ORDER BY u.person_id
;
