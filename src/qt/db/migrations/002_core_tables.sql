-- 002: every table from the spec's schema section, plus the changes in
-- docs/components/01_infrastructure.md section 4.2. All timestamps are timestamptz (UTC).

-- ------------------------------------------------------------------ reference
CREATE TABLE assets (
    asset_id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticker           text        NOT NULL,
    asset_class      text        NOT NULL
        CHECK (asset_class IN ('equity', 'etf', 'crypto', 'index', 'future', 'fx', 'other')),
    name             text,
    cik              text,
    alpaca_id        uuid UNIQUE,
    gics_sector      text,
    ipo_date         date,
    alpaca_tradable  boolean     NOT NULL DEFAULT false,
    created_at       timestamptz NOT NULL DEFAULT now(),
    updated_at       timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX assets_ticker_idx ON assets (ticker);

CREATE TABLE universe_weekly (
    week           date    NOT NULL,
    asset_id       bigint  NOT NULL REFERENCES assets,
    tier_eligible  boolean NOT NULL,
    factor_ranks   jsonb   NOT NULL DEFAULT '{}',
    run_id         text,
    PRIMARY KEY (week, asset_id)
);

-- ------------------------------------------------------------------ raw (append-only, see 004)
CREATE TABLE raw_prices (
    asset_id     bigint           NOT NULL REFERENCES assets,
    timeframe    text             NOT NULL DEFAULT '1d',
    ts           timestamptz      NOT NULL,          -- bar start
    open         double precision,
    high         double precision,
    low          double precision,
    close        double precision,
    volume       double precision,
    vwap         double precision,
    trade_count  bigint,
    source       text             NOT NULL,
    released_at  timestamptz      NOT NULL,          -- bar close: when the bar became knowable
    ingested_at  timestamptz      NOT NULL DEFAULT now(),
    UNIQUE (asset_id, timeframe, ts, source)
);

CREATE TABLE raw_macro (
    series_id    text             NOT NULL,
    period       date             NOT NULL,
    value        double precision,                   -- null = source reported missing
    vintage      date             NOT NULL,          -- ALFRED realtime_start
    released_at  timestamptz      NOT NULL,
    source       text             NOT NULL DEFAULT 'fred',
    ingested_at  timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (series_id, period, vintage)
);
CREATE INDEX raw_macro_released_idx ON raw_macro (series_id, released_at);

CREATE TABLE raw_fundamentals (
    asset_id       bigint           NOT NULL REFERENCES assets,
    fiscal_period  text             NOT NULL,
    metric         text             NOT NULL,
    value          double precision,
    unit           text,
    form           text,
    accession      text             NOT NULL,
    filed_at       timestamptz      NOT NULL,
    released_at    timestamptz      NOT NULL,        -- = EDGAR acceptance time
    ingested_at    timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (asset_id, fiscal_period, metric, accession)
);
CREATE INDEX raw_fundamentals_released_idx ON raw_fundamentals (asset_id, metric, released_at);

-- ------------------------------------------------------------------ news
CREATE TABLE articles (
    article_id    bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    url           text        NOT NULL,
    url_hash      text        NOT NULL UNIQUE,
    simhash       bigint,
    source        text        NOT NULL,
    published_at  timestamptz,
    released_at   timestamptz NOT NULL,              -- published_at, or first-seen if missing
    title         text,
    body          text,
    tickers       text[]      NOT NULL DEFAULT '{}',
    ingested_at   timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX articles_released_idx ON articles (released_at);

-- svm_prob / kept live here so articles stays immutable (component doc 4.4).
CREATE TABLE article_scores (
    article_id  bigint           NOT NULL REFERENCES articles,
    run_id      text             NOT NULL,
    svm_prob    double precision CHECK (svm_prob BETWEEN 0 AND 1),
    kept        boolean          NOT NULL,
    scored_at   timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (article_id, run_id)
);

CREATE TABLE article_seen (
    url_hash    text        PRIMARY KEY,
    simhash     bigint,
    first_seen  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE llm_outputs (
    run_id      text             NOT NULL,
    article_id  bigint           NOT NULL REFERENCES articles,
    keywords    text[]           NOT NULL DEFAULT '{}',
    sentiment   double precision,
    event_type  text,
    confidence  double precision CHECK (confidence BETWEEN 0 AND 1),
    created_at  timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (run_id, article_id)
);

CREATE TABLE ngram_vocab (
    version   int              NOT NULL,
    ngram     text             NOT NULL,
    chi2      double precision NOT NULL,
    built_at  timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (version, ngram)
);

-- ------------------------------------------------------------------ features
CREATE TABLE feature_registry (
    feature_name   text        NOT NULL,
    version        int         NOT NULL,
    formula        text        NOT NULL,
    "window"       text,
    source_series  text[]      NOT NULL DEFAULT '{}',
    created_at     timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (feature_name, version)
);

CREATE TABLE features (
    asof             timestamptz      NOT NULL,
    asset_id         bigint REFERENCES assets,       -- null = market-level feature
    feature_name     text             NOT NULL,
    feature_version  int              NOT NULL,
    value            double precision,
    computed_at      timestamptz      NOT NULL DEFAULT now(),
    run_id           text
);
CREATE UNIQUE INDEX features_key_idx
    ON features (asof, asset_id, feature_name, feature_version) NULLS NOT DISTINCT;

CREATE TABLE regime (
    asof          timestamptz      PRIMARY KEY,
    quadrant      text,
    hmm_calm      double precision,
    hmm_normal    double precision,
    hmm_stress    double precision,
    stress_score  double precision,
    run_id        text
);

-- ------------------------------------------------------------------ models
CREATE TABLE predictions (
    asof           timestamptz      NOT NULL,
    run_id         text             NOT NULL,
    model_name     text             NOT NULL,
    model_version  text             NOT NULL,
    asset_id       bigint           NOT NULL REFERENCES assets,
    p_up           double precision CHECK (p_up   BETWEEN 0 AND 1),
    p_flat         double precision CHECK (p_flat BETWEEN 0 AND 1),
    p_down         double precision CHECK (p_down BETWEEN 0 AND 1),
    created_at     timestamptz      NOT NULL DEFAULT now(),
    UNIQUE (asof, run_id, model_name, asset_id)
);

CREATE TABLE meta_scores (
    run_id       text             NOT NULL,
    asset_id     bigint           NOT NULL REFERENCES assets,
    asof         timestamptz      NOT NULL,
    score        double precision,
    conviction   double precision,
    market_risk  double precision,
    PRIMARY KEY (run_id, asset_id)
);

-- ------------------------------------------------------------------ portfolio / execution
CREATE TABLE positions (
    asset_id    bigint      PRIMARY KEY REFERENCES assets,
    qty         numeric     NOT NULL,
    avg_cost    numeric     NOT NULL,
    tier        text,
    opened_at   timestamptz NOT NULL,
    peak_price  numeric,
    stop_price  numeric,
    updated_at  timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE portfolio_daily (
    date            date    PRIMARY KEY,
    nav             numeric NOT NULL,
    cash            numeric NOT NULL,
    sleeve_weights  jsonb   NOT NULL DEFAULT '{}',
    mode            text,
    drawdown        double precision,
    run_id          text
);

CREATE TABLE proposed_orders (
    proposed_id     bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_id          text        NOT NULL,
    asset_id        bigint      NOT NULL REFERENCES assets,
    side            text        NOT NULL CHECK (side IN ('buy', 'sell')),
    target_w        double precision,
    delta_notional  numeric,
    tier            text,
    reason_code     text        NOT NULL,
    rule_checks     jsonb       NOT NULL DEFAULT '{}',
    status          text        NOT NULL DEFAULT 'proposed',
    created_at      timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE orders (
    order_id      bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    alpaca_id     uuid UNIQUE,
    proposed_id   bigint REFERENCES proposed_orders,
    asset_id      bigint      NOT NULL REFERENCES assets,
    side          text        NOT NULL CHECK (side IN ('buy', 'sell')),
    qty           numeric,
    limit_price   numeric,
    submitted_at  timestamptz,
    status        text        NOT NULL,
    filled_qty    numeric     NOT NULL DEFAULT 0,
    avg_fill      numeric,
    slippage_bps  double precision,
    updated_at    timestamptz NOT NULL DEFAULT now()
);

-- ------------------------------------------------------------------ ops
CREATE TABLE runs (
    run_id          text        PRIMARY KEY,
    stage           text        NOT NULL,
    profile         text        NOT NULL,
    config_hash     text        NOT NULL,
    git_sha         text        NOT NULL,
    started_at      timestamptz NOT NULL DEFAULT now(),
    finished_at     timestamptz,
    status          text        NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'ok', 'failed')),
    feed_freshness  jsonb       NOT NULL DEFAULT '{}',
    model_versions  jsonb       NOT NULL DEFAULT '{}',
    error           text
);
CREATE INDEX runs_started_idx ON runs (started_at);

CREATE TABLE risk_events (
    id       bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts       timestamptz NOT NULL DEFAULT now(),
    type     text        NOT NULL,
    details  jsonb       NOT NULL DEFAULT '{}',
    run_id   text
);

CREATE TABLE alerts_log (
    id         bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts         timestamptz NOT NULL DEFAULT now(),
    severity   text        NOT NULL CHECK (severity IN ('info', 'warn', 'critical')),
    event      text        NOT NULL,
    title      text        NOT NULL,
    body       text,
    delivered  boolean     NOT NULL,
    run_id     text
);
CREATE INDEX alerts_log_ts_idx ON alerts_log (ts);

CREATE TABLE backup_log (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ts          timestamptz NOT NULL DEFAULT now(),
    run_id      text,
    dump_file   text,
    dump_bytes  bigint,
    status      text        NOT NULL CHECK (status IN ('ok', 'failed')),
    row_counts  jsonb       NOT NULL DEFAULT '{}',
    mismatches  jsonb       NOT NULL DEFAULT '{}',
    error       text
);

-- ------------------------------------------------------------------ views
-- Latest value of every feature per asset (null asset = market). Phase 3 refines this.
CREATE VIEW model_input_latest AS
SELECT DISTINCT ON (asset_id, feature_name)
       asset_id, feature_name, feature_version, asof, value
FROM features
ORDER BY asset_id, feature_name, asof DESC, feature_version DESC;
