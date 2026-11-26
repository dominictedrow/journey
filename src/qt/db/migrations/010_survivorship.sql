-- 010: Phase 3 build step 5c (docs/components/03_features.md section 9, D-043).

-- Sectors of non-current members come from the GICS sector in the Wikipedia constituents table
-- as it stood while they were members (an old page revision), else from their SIC code.
ALTER TABLE assets DROP CONSTRAINT assets_sector_source_check;
ALTER TABLE assets ADD CONSTRAINT assets_sector_source_check
    CHECK (sector_source IN ('wikipedia_gics', 'wikipedia_gics_hist', 'sic_map'));

-- Members added since 008 got a Wikipedia sector without a source label.
UPDATE assets SET sector_source = 'wikipedia_gics'
WHERE gics_sector IS NOT NULL AND sector_source IS NULL;
