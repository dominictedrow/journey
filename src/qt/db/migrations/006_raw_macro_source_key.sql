-- 006: raw_macro primary key includes source (D-029).
-- ALFRED vintages (source 'alfred') and the HY OAS archive (source 'fred_archive') can share a
-- (series_id, period, vintage): the archive's vintage is the fetch date, and FRED stamps
-- BAMLH0A0HYM2 vintages with the observation date. Without source in the key, the second row
-- would be dropped by ON CONFLICT DO NOTHING. ALTER is allowed: the append-only triggers block
-- UPDATE/DELETE/TRUNCATE, not DDL.
ALTER TABLE raw_macro DROP CONSTRAINT raw_macro_pkey;
ALTER TABLE raw_macro ADD PRIMARY KEY (series_id, source, period, vintage);
