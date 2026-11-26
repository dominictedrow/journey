-- 007: asset identity for index members across renames, delistings and ticker reuse (D-030).
-- security_key: stable id of an S&P 500 member security, "SPX:<first ticker>:<first date>";
--   null for ETFs/crypto (identified by alpaca_id).
-- listed_until: last trading day for a security that stopped trading (delisted, acquired, or its
--   symbol went quiet before being reused); null while it still trades.
ALTER TABLE assets ADD COLUMN security_key text UNIQUE;
ALTER TABLE assets ADD COLUMN listed_until date;
