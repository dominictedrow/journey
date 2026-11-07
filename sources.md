# Sources

## All Websites Mentioned in framework.md

1. lynalden.com
2. federalreserve.gov
3. fred.stlouisfed.org
4. cboe.com/vix
5. tradingview.com
6. lme.com
7. spglobal.com/pmi
8. data.oecd.org
9. tradingeconomics.com
10. ecb.europa.eu
11. boj.or.jp
12. imf.org/en/Publications/WEO
13. imf.org/en/Publications/FM
14. cfr.org/global-conflict-tracker
15. etf.com
16. cftc.gov/COT
17. pages.stern.nyu.edu/~adamodar
18. lookintobitcoin.com
19. farside.co.uk/bitcoin-etf-flow
20. defillama.com
21. alternative.me/crypto/fear-and-greed-index
22. coinglass.com
23. macrotrends.net
24. tikr.com
25. stockanalysis.com
26. finviz.com
27. koyfin.com
28. seekingalpha.com
29. tipranks.com
30. bls.gov
31. ismworld.org
32. advantage.factset.com
33. stockcharts.com
34. blockchaincenter.net
35. starcapital.de/en/research
36. cmegroup.com

---

## Programmatically Integrable Links (APIs & Data Endpoints)

These endpoints can be called directly from code without authentication unless noted.

### Free Public APIs (no key required)

| Source | Endpoint | Data |
|--------|----------|------|
| Alternative.me | `https://api.alternative.me/fng/` | Crypto Fear & Greed Index (JSON) |
| Alternative.me (history) | `https://api.alternative.me/fng/?limit=30` | Last 30 days Fear & Greed |
| DefiLlama | `https://stablecoins.llama.fi/stablecoins` | Stablecoin market caps (JSON) |
| DefiLlama TVL | `https://api.llama.fi/v2/chains` | TVL by chain (JSON) |
| IMF DataMapper | `https://www.imf.org/external/datamapper/api/v1/` | GDP, inflation, fiscal data (JSON) |
| CFTC COT Data | `https://www.cftc.gov/dea/newcot/c_disagg.txt` | Commitments of Traders (CSV) |

### API Key Required (free tier available)

| Source | Base URL | Key Data | Notes |
|--------|----------|----------|-------|
| FRED (St. Louis Fed) | `https://api.stlouisfed.org/fred/series/observations` | T10Y2Y, BAMLH0A0HYM2, M2SL, DGS10, DFII10 | Free API key at fred.stlouisfed.org |
| BLS | `https://api.bls.gov/publicAPI/v2/timeseries/data/` | CPI (CUUR0000SA0), PPI, Employment | Free key at bls.gov/developers |

### Key FRED Series IDs for Integration

```
T10Y2Y        — 2s/10s yield curve spread
BAMLH0A0HYM2  — HY credit spread (OAS)
M2SL          — US M2 money supply
DGS10         — 10-year Treasury yield
DFII10        — 10-year TIPS (real yield)
DEXJPUS       — USD/JPY exchange rate
DEXUSAL       — AUD/USD exchange rate
```

### Key BLS Series IDs for Integration

```
CUUR0000SA0   — CPI All Urban Consumers (All Items)
WPSFD4        — PPI Final Demand
CES0000000001 — Total Nonfarm Payrolls
```

### Web Scraping Targets (no API, structured HTML/JSON embedded)

| Source | URL | Data Available |
|--------|-----|----------------|
| Farside Bitcoin ETF Flows | `https://farside.co.uk/bitcoin-etf-flow/` | Daily ETF net flows table |
| Blockchaincenter Altcoin Index | `https://blockchaincenter.net/altcoin-season-index/` | Altcoin season score |
| LookIntoBitcoin MVRV | `https://lookintobitcoin.com/charts/mvrv-zscore/` | MVRV Z-Score chart data |

### Paid / Rate-Limited (integration possible with subscription)

| Source | Notes |
|--------|-------|
| tradingview.com | Charting library available; raw data requires paid plan |
| coinglass.com | Public API at `https://open-api.coinglass.com` — free tier limited |
| finviz.com | Screener exports require Elite plan |
| seekingalpha.com | API available but paid |
| tipranks.com | API available but paid |
| spglobal.com/pmi | PMI data requires subscription |
| lme.com | Metals data requires subscription |
| stockanalysis.com | Some free endpoints; bulk data paid |
