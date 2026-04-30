-- 00_ingest_profile.sql
-- Profile data/raw/events.parquet across 9 dimensions.
-- Run via sql_runner.run_sql_file(); returns a single labelled result set.

WITH
  src AS (
    SELECT * FROM read_parquet('data/raw/events.parquet')
  ),

  -- 1. Row count
  q1 AS (
    SELECT '01_total_rows' AS metric, COUNT(*)::VARCHAR AS value FROM src
  ),

  -- 2. Distinct users
  q2 AS (
    SELECT '02_distinct_users' AS metric, COUNT(DISTINCT person_id)::VARCHAR AS value FROM src
  ),

  -- 3. Distinct event types
  q3 AS (
    SELECT '03_distinct_event_types' AS metric, COUNT(DISTINCT event)::VARCHAR AS value FROM src
  ),

  -- 4. Date range (min and max timestamp)
  q4 AS (
    SELECT '04_timestamp_min' AS metric, MIN(timestamp)::VARCHAR AS value FROM src
    UNION ALL
    SELECT '04_timestamp_max' AS metric, MAX(timestamp)::VARCHAR AS value FROM src
  ),

  -- 5. Positive class size: users with subscription_upgraded
  q5 AS (
    SELECT '05_users_with_subscription_upgraded' AS metric,
           COUNT(DISTINCT person_id)::VARCHAR AS value
    FROM src
    WHERE event = 'subscription_upgraded'
  ),

  -- 6. Positive rate (upgraded / total distinct users)
  q6 AS (
    SELECT '06_positive_rate_pct' AS metric,
           ROUND(
             100.0 * COUNT(DISTINCT CASE WHEN event = 'subscription_upgraded' THEN person_id END)
             / NULLIF(COUNT(DISTINCT person_id), 0),
           2)::VARCHAR AS value
    FROM src
  ),

  -- 7. Null / missing counts for key columns
  q7 AS (
    SELECT '07_null_person_id'   AS metric, COUNT(*)::VARCHAR AS value FROM src WHERE person_id IS NULL
    UNION ALL
    SELECT '07_null_event'       AS metric, COUNT(*)::VARCHAR AS value FROM src WHERE event     IS NULL
    UNION ALL
    SELECT '07_null_timestamp'   AS metric, COUNT(*)::VARCHAR AS value FROM src WHERE timestamp  IS NULL
  ),

  -- 8. Top-10 events by frequency
  q8 AS (
    SELECT '08_top_event_' || ROW_NUMBER() OVER (ORDER BY cnt DESC) AS metric,
           event || ' (' || cnt::VARCHAR || ')' AS value
    FROM (
      SELECT event, COUNT(*) AS cnt FROM src GROUP BY event ORDER BY cnt DESC LIMIT 10
    ) t
  ),

  -- 9. Count of distinct users with promo_code_redeemed
  --    (replaces the old "redeem upgrade offer" check)
  q9 AS (
    SELECT '09_users_promo_code_redeemed' AS metric,
           COUNT(DISTINCT person_id)::VARCHAR AS value
    FROM src
    WHERE event = 'promo_code_redeemed'
  )

SELECT metric, value FROM q1
UNION ALL SELECT metric, value FROM q2
UNION ALL SELECT metric, value FROM q3
UNION ALL SELECT metric, value FROM q4
UNION ALL SELECT metric, value FROM q5
UNION ALL SELECT metric, value FROM q6
UNION ALL SELECT metric, value FROM q7
UNION ALL SELECT metric, value FROM q8
UNION ALL SELECT metric, value FROM q9
ORDER BY metric
;
