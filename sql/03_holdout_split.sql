-- sql/03_holdout_split.sql
-- Time-based train/val/test split.
-- Reads from data/processed/labels.parquet (produced by 02_label_generation.sql).
-- Splits by user_cutoff_ts percentiles: train <= P70, val (P70, P85], test > P85.
-- Writes person_id + split column.

WITH labeled AS (
    SELECT person_id, will_upgrade_in_7d, user_cutoff_ts
    FROM read_parquet('data/processed/labels.parquet')
),
cutoffs AS (
    SELECT
        QUANTILE_CONT(EPOCH(user_cutoff_ts::TIMESTAMP), 0.70) AS p70_epoch,
        QUANTILE_CONT(EPOCH(user_cutoff_ts::TIMESTAMP), 0.85) AS p85_epoch
    FROM labeled
)
SELECT
    l.person_id,
    l.will_upgrade_in_7d,
    l.user_cutoff_ts,
    CASE
        WHEN EPOCH(l.user_cutoff_ts::TIMESTAMP) <= c.p70_epoch THEN 'train'
        WHEN EPOCH(l.user_cutoff_ts::TIMESTAMP) <= c.p85_epoch THEN 'validation'
        ELSE 'test'
    END AS split
FROM labeled l
CROSS JOIN cutoffs c;
