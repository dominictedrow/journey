-- 004: raw tables are append-only (CLAUDE.md hard rule). UPDATE / DELETE / TRUNCATE raise.
CREATE FUNCTION qt_block_mutation() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'append-only table "%": % is not allowed', TG_TABLE_NAME, TG_OP
        USING ERRCODE = 'insufficient_privilege';
END;
$$;

DO $$
DECLARE
    t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['raw_prices', 'raw_macro', 'raw_fundamentals', 'articles', 'article_seen']
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
