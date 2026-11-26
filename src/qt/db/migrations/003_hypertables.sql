-- 003: hypertables + compression (D-012). Initial values mirror configs/db.yaml;
-- later changes go through a new migration.
SELECT create_hypertable('raw_prices',  by_range('ts',   INTERVAL '30 days'));
SELECT create_hypertable('features',    by_range('asof', INTERVAL '30 days'));
SELECT create_hypertable('predictions', by_range('asof', INTERVAL '90 days'));

ALTER TABLE raw_prices SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'asset_id, timeframe',
    timescaledb.compress_orderby   = 'ts DESC'
);
ALTER TABLE features SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'feature_name, asset_id',
    timescaledb.compress_orderby   = 'asof DESC'
);
ALTER TABLE predictions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'model_name',
    timescaledb.compress_orderby   = 'asof DESC'
);

SELECT add_compression_policy('raw_prices',  INTERVAL '90 days');
SELECT add_compression_policy('features',    INTERVAL '90 days');
SELECT add_compression_policy('predictions', INTERVAL '180 days');
