-- 02_label_generation.sql
-- Build binary labels for churn / upgrade prediction.
--
-- DESIGN DECISIONS
-- ─────────────────
-- • Positive class  : user fired 'subscription_upgraded'.
-- • Negative class  : user never upgraded AND never redeemed a promo code.
--   (promo_code_redeemed users are excluded to avoid contaminating the
--    negative class with users who may have been close to upgrading.)
-- • MIN_EVENTS_FOR_INCLUSION = 5
--   Both positives and negatives must have ≥5 pre-cutoff clean events.
-- • The 'redeem upgrade offer' event does not exist in this dataset
--   and has been dropped entirely.

WITH

  -- ── Raw source ───────────────────────────────────────────────────────────
  raw AS (
    SELECT * FROM read_parquet('data/raw/events.parquet')
  ),

  -- ── Cutoff: upgrade ts for positives, MAX ts for everyone else ───────────
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

  -- ── Clean events (same banned list as feature_store) ────────────────────
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

  -- ── Pre-cutoff event counts ──────────────────────────────────────────────
  event_counts AS (
    SELECT person_id, COUNT(*) AS pre_cutoff_events
    FROM clean_events
    GROUP BY person_id
  ),

  -- ── Positive class (before MIN filter) ──────────────────────────────────
  pos_raw AS (
    SELECT DISTINCT person_id, 1 AS will_upgrade_in_7d
    FROM raw
    WHERE event = 'subscription_upgraded'
  ),

  -- ── Promo-code users (excluded from negatives) ───────────────────────────
  promo_users AS (
    SELECT DISTINCT person_id
    FROM raw
    WHERE event = 'promo_code_redeemed'
  ),

  -- ── Negative class (before MIN filter) ──────────────────────────────────
  neg_raw AS (
    SELECT DISTINCT r.person_id, 0 AS will_upgrade_in_7d
    FROM (SELECT DISTINCT person_id FROM raw) r
    LEFT JOIN pos_raw  p USING (person_id)
    LEFT JOIN promo_users pu USING (person_id)
    WHERE p.person_id  IS NULL   -- not a positive
      AND pu.person_id IS NULL   -- not a promo redeemer
  ),

  -- ── Combine before MIN_EVENTS filter ────────────────────────────────────
  combined_raw AS (
    SELECT * FROM pos_raw
    UNION ALL
    SELECT * FROM neg_raw
  ),

  -- ── Apply MIN_EVENTS_FOR_INCLUSION = 5 to BOTH classes ──────────────────
  labeled AS (
    SELECT c.person_id, c.will_upgrade_in_7d, uc.user_cutoff_ts
    FROM combined_raw c
    JOIN event_counts ec USING (person_id)
    JOIN user_cutoff  uc USING (person_id)
    WHERE ec.pre_cutoff_events >= 5
  ),

  -- ── Counts for diagnostics (printed by the caller via sql_runner) ────────
  diag AS (
    SELECT
      COUNT(CASE WHEN will_upgrade_in_7d = 1 THEN 1 END) AS positives_after,
      COUNT(CASE WHEN will_upgrade_in_7d = 0 THEN 1 END) AS negatives_after,
      (SELECT COUNT(*) FROM pos_raw)                      AS positives_before,
      (SELECT COUNT(*) FROM neg_raw)                      AS negatives_before
    FROM labeled
  )

-- ── Final output: labeled set + diagnostics appended as a summary row ────
SELECT
  person_id,
  will_upgrade_in_7d,
  user_cutoff_ts,
  NULL::BIGINT AS _diag_positives_before,
  NULL::BIGINT AS _diag_positives_after,
  NULL::BIGINT AS _diag_negatives_before,
  NULL::BIGINT AS _diag_negatives_after
FROM labeled

UNION ALL

SELECT
  NULL AS person_id,
  NULL AS will_upgrade_in_7d,
  NULL AS user_cutoff_ts,
  positives_before  AS _diag_positives_before,
  positives_after   AS _diag_positives_after,
  negatives_before  AS _diag_negatives_before,
  negatives_after   AS _diag_negatives_after
FROM diag

ORDER BY person_id NULLS LAST
;
