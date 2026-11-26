-- 009: Phase 3 build step 5b (docs/components/03_features.md sections 8 and 10, D-042).

-- Compression layout for the features history (~200M rows from 2016). Segmenting by
-- (feature_name, asset_id) left ~21 rows per segment in a 30-day chunk, which barely compresses;
-- (feature_name, feature_version) gives ~560 assets x 21 days per segment. The backfill
-- compresses each chunk once it is complete. Only allowed while no chunk is compressed.
SELECT remove_compression_policy('features', if_exists => true);
ALTER TABLE features SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'feature_name, feature_version',
    timescaledb.compress_orderby   = 'asset_id, asof DESC'
);
SELECT add_compression_policy('features', INTERVAL '90 days');

-- features_daily marks a run whose critical inputs were stale instead of computing (section 10).
ALTER TABLE runs DROP CONSTRAINT runs_status_check;
ALTER TABLE runs ADD CONSTRAINT runs_status_check
    CHECK (status IN ('running', 'ok', 'failed', 'stale_inputs'));
