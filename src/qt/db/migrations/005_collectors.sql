-- 005: Phase 2 collector tables (docs/components/02_collectors.md section 4, D-026).
-- Raw tables are append-only (triggers at the bottom). Every raw row carries released_at NOT NULL.

-- ------------------------------------------------------------------ membership
CREATE TABLE index_membership (
    index_code   text        NOT NULL,                -- 'SPX'
    snapshot_at  timestamptz NOT NULL,
    ticker       text        NOT NULL,
    source       text        NOT NULL,                -- 'fja05680' | 'wikipedia'
    asset_id     bigint REFERENCES assets,            -- null until mapped
    released_at  timestamptz NOT NULL,
    ingested_at  timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (index_code, snapshot_at, ticker, source)
);
CREATE INDEX index_membership_released_idx ON index_membership (index_code, released_at);

-- ------------------------------------------------------------------ aliases (reference, mutable)
CREATE TABLE aliases (
    alias_id        bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    asset_id        bigint  NOT NULL REFERENCES assets,
    alias           text    NOT NULL,
    kind            text    NOT NULL CHECK (kind IN ('edgar_name', 'short', 'manual', 'ticker')),
    case_sensitive  boolean NOT NULL DEFAULT false,
    needs_cue       boolean NOT NULL DEFAULT false,
    valid_from      date,
    valid_to        date,
    source          text    NOT NULL
);
CREATE UNIQUE INDEX aliases_key_idx
    ON aliases (asset_id, alias, kind, valid_from) NULLS NOT DISTINCT;
CREATE INDEX aliases_alias_idx ON aliases (lower(alias));

-- ------------------------------------------------------------------ events
-- Fed items, FOMC dates, Congress actions, earnings dates. A changed item is a new row.
CREATE TABLE raw_events (
    source        text        NOT NULL,
    external_id   text        NOT NULL,
    content_hash  text        NOT NULL,
    kind          text        NOT NULL,
    event_ts      timestamptz,
    title         text,
    url           text,
    payload       jsonb       NOT NULL DEFAULT '{}',
    released_at   timestamptz NOT NULL,
    ingested_at   timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (source, external_id, content_hash)
);
CREATE INDEX raw_events_kind_released_idx ON raw_events (kind, released_at);

-- ------------------------------------------------------------------ generic metrics
-- CoinGecko volume (subject = coin), BTC dominance, stablecoin supply, ETF flows ('' = market).
CREATE TABLE raw_metrics (
    source       text             NOT NULL,
    metric       text             NOT NULL,
    subject      text             NOT NULL DEFAULT '',
    ts           timestamptz      NOT NULL,
    value        double precision,
    unit         text,
    released_at  timestamptz      NOT NULL,
    ingested_at  timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (source, metric, subject, ts)
);
CREATE INDEX raw_metrics_released_idx ON raw_metrics (metric, subject, released_at);

-- ------------------------------------------------------------------ crypto derivatives
CREATE TABLE raw_crypto_derivs (
    venue               text             NOT NULL,
    symbol              text             NOT NULL,
    ts                  timestamptz      NOT NULL,
    kind                text             NOT NULL CHECK (kind IN ('funding', 'oi')),
    funding_rate        double precision,
    funding_interval_h  double precision,
    oi_contracts        double precision,
    oi_usd              double precision,
    mark_price          double precision,
    released_at         timestamptz      NOT NULL,
    ingested_at         timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (venue, symbol, ts, kind)
);
CREATE INDEX raw_crypto_derivs_released_idx ON raw_crypto_derivs (symbol, kind, released_at);

-- ------------------------------------------------------------------ ops (mutable)
CREATE TABLE collector_state (
    collector        text        PRIMARY KEY,
    last_ok_at       timestamptz,
    last_attempt_at  timestamptz,
    last_status      text CHECK (last_status IN ('ok', 'failed', 'skipped')),
    last_error       text,
    cursor           jsonb       NOT NULL DEFAULT '{}',
    rows_last_run    bigint
);

-- ------------------------------------------------------------------ articles (additive only)
ALTER TABLE articles
    ADD COLUMN word_count      int,
    ADD COLUMN extract_status  text CHECK (extract_status IN ('ok', 'short', 'failed', 'skipped')),
    ADD COLUMN guid            text,
    ADD COLUMN title_hash      text,
    ADD COLUMN near_dup_of     bigint REFERENCES articles;
CREATE INDEX articles_title_hash_idx ON articles (title_hash);

COMMENT ON COLUMN articles.released_at IS
    'First-seen time: when our fetch inserted the article (D-022). published_at is the '
    'publisher''s claim, stored separately and never used for point-in-time filtering.';

-- ------------------------------------------------------------------ append-only triggers
DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['index_membership', 'raw_events', 'raw_metrics', 'raw_crypto_derivs']
    LOOP
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE UPDATE OR DELETE ON %I FOR EACH ROW EXECUTE FUNCTION qt_block_mutation()',
            t || '_append_only', t);
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE TRUNCATE ON %I FOR EACH STATEMENT EXECUTE FUNCTION qt_block_mutation()',
            t || '_no_truncate', t);
    END LOOP;
END;
$$;
