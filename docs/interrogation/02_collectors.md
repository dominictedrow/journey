# Questioning: Phase 2 · Collectors (point-in-time)

> Ask Dom **one question at a time**. Record each answer under the question as `**A:**`. When every question is answered, write `docs/components/02_collectors.md`, covering interfaces, config keys, tests and acceptance criteria.
> Spec reference: Data collection, Point-in-time rules

1. **Universe:** what does the weekly screen start from? Options: the S&P 500, the S&P 500 plus the Nasdaq-100, or the Russell 1000. How big should the tradable universe be (the spec implies about 60 stock candidates)?
2. **Crypto universe:** BTC, ETH and SOL plus which altcoins? Should it be every Alpaca-tradable coin that passes the $50M ADV filter?
3. **Backfill depth:** 10 years (2016 onward, the limit of Alpaca's free history), or go further back with yfinance?
4. **ALFRED:** should training use first-release vintages for every macro series, or only for the revised ones (CPI, PCE, GDP, UNRATE)?
5. **Historical S&P membership (fja05680):** before our first stored screen, OK to accept and document its known gaps in the early years?
6. **News:** collect every article from the 5 feeds now, before the quota and SVM exist? (Recommended: yes. Store everything; filtering happens downstream.) How often should we fetch: hourly or every run?
7. **Aliases table:** how should the ticker → company names and short-names list be built (EDGAR company names plus manual additions)?
8. **Congress.gov:** which committees and bill statuses count as "finance, tax and appropriations"? Should Fed speeches be included?
9. **Crypto derivatives (D-006):** which symbols' funding and open interest (BTC and ETH perps only)? How should venues be combined (OI-weighted average)?
10. **SoSoValue (D-007):** if its key turns out to be free, use it. If not, drop ETF flows. Any objection?
11. **Daily job timing:** confirm the 16:30 ET collect. Crypto trades 24/7, so which cutoff defines the crypto "daily close" (00:00 UTC, or 16:00 ET)?
12. **EDGAR User-Agent:** what app name and contact string should we send (kept in `.env`)?
