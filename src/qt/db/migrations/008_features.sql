-- 008: Phase 3 feature layer (docs/components/03_features.md, D-033).

-- Registry: a (feature_name, version) row is immutable in what it computes; only its lifecycle
-- (status, changelog) may change. A formula change must arrive as a new version.
ALTER TABLE feature_registry ADD COLUMN feature_group text NOT NULL DEFAULT 'misc';
ALTER TABLE feature_registry ADD COLUMN changelog text;
ALTER TABLE feature_registry ADD COLUMN status text NOT NULL DEFAULT 'active'
    CHECK (status IN ('active', 'frozen'));

CREATE FUNCTION qt_registry_immutable() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'feature_registry rows cannot be deleted (freeze instead)'
            USING ERRCODE = 'insufficient_privilege';
    END IF;
    IF NEW.feature_name IS DISTINCT FROM OLD.feature_name
       OR NEW.version IS DISTINCT FROM OLD.version
       OR NEW.formula IS DISTINCT FROM OLD.formula
       OR NEW."window" IS DISTINCT FROM OLD."window"
       OR NEW.source_series IS DISTINCT FROM OLD.source_series
       OR NEW.feature_group IS DISTINCT FROM OLD.feature_group THEN
        RAISE EXCEPTION 'feature_registry (%, v%): definition is immutable; bump the version',
            OLD.feature_name, OLD.version USING ERRCODE = 'insufficient_privilege';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER feature_registry_immutable BEFORE UPDATE OR DELETE ON feature_registry
    FOR EACH ROW EXECUTE FUNCTION qt_registry_immutable();

-- Feature values are append-only, like the raw tables (Q7): a new version is written alongside.
CREATE TRIGGER features_append_only BEFORE UPDATE OR DELETE ON features
    FOR EACH ROW EXECUTE FUNCTION qt_block_mutation();
CREATE TRIGGER features_no_truncate BEFORE TRUNCATE ON features
    FOR EACH STATEMENT EXECUTE FUNCTION qt_block_mutation();

-- Regime rows are versioned and append-only as well, so a fix ships as a new regime_version.
ALTER TABLE regime ADD COLUMN regime_version int NOT NULL DEFAULT 1;
ALTER TABLE regime DROP CONSTRAINT regime_pkey;
ALTER TABLE regime ADD PRIMARY KEY (asof, regime_version);
CREATE TRIGGER regime_append_only BEFORE UPDATE OR DELETE ON regime
    FOR EACH ROW EXECUTE FUNCTION qt_block_mutation();
CREATE TRIGGER regime_no_truncate BEFORE TRUNCATE ON regime
    FOR EACH STATEMENT EXECUTE FUNCTION qt_block_mutation();

-- One row per monthly HMM refit (Q5): reproducible from seed + window.
CREATE TABLE hmm_fits (
    fit_date        date             NOT NULL,
    regime_version  int              NOT NULL DEFAULT 1,
    train_start     date             NOT NULL,
    train_end       date             NOT NULL,
    n_weeks         int              NOT NULL,
    best_seed       int              NOT NULL,
    log_likelihood  double precision NOT NULL,
    state_order     int[]            NOT NULL,   -- fitted state index for calm, normal, stress
    relabel_change  double precision,            -- share of prior 52 weeks whose state changed
    model_path      text             NOT NULL,
    run_id          text,
    created_at      timestamptz      NOT NULL DEFAULT now(),
    PRIMARY KEY (fit_date, regime_version)
);

-- Where an asset's sector label came from (Q3).
ALTER TABLE assets ADD COLUMN sector_source text
    CHECK (sector_source IN ('wikipedia_gics', 'sic_map'));
UPDATE assets SET sector_source = 'wikipedia_gics' WHERE gics_sector IS NOT NULL;
ALTER TABLE assets ADD COLUMN sic text;

-- model_input_latest predates versioning and mixes versions; matrix.features_asof replaces it.
DROP VIEW model_input_latest;
