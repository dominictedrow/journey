# API Registry

Verified by research on **2026-11-25** from a US IP. Re-verify anything marked ⚠️ when its collector is built.

Legend: ✅ free, no key · 🔑 free with a key · ⚠️ fragile, unofficial or unverified · ❌ **PAID (not used)**

## In use
| Data | Source | Status | Auth | Limits | Notes / fallback |
|---|---|---|---|---|---|
| Stock/ETF daily and minute bars | Alpaca Market Data (Basic) | 🔑 | Alpaca key | 200 req/min; history since 2016 | Real-time is the IEX feed only. SIP data older than 15 minutes is available for history. Fallbacks: yfinance, Stooq |
| Crypto bars and L2 orderbook | Alpaca `/v1beta3/crypto/us/...` | 🔑 ⚠️ | Alpaca key | Crypto limits not documented | Fallback: Kraken or Coinbase public REST |
| Assets (tradability) | Alpaca Trading API | 🔑 | Paper key | | Check every run (`asset_class=crypto, tradable=true`) |
| Paper trading | `paper-api.alpaca.markets` | 🔑 | Paper key (`PK…`) | | $100k default. No slippage model. About 10% of fills are random partial fills |
| SPY, NQ=F, GC=F, DX-Y.NYB, ^VIX, ^VIX3M | yfinance | ⚠️ | none | IP-based 429s | Unofficial. Cache everything. VIX fallback: FRED `VIXCLS` |
| Yields, rates, inflation, labor, GDP, liquidity series | FRED | 🔑 | FRED key | ~120 req/min | Series listed in the spec. See the HY OAS caveat below |
| Macro vintages (first-release values) | ALFRED (same API) | 🔑 | FRED key | same | `realtime_start` / `vintage_dates` |
| HY credit spread | FRED `BAMLH0A0HYM2` | ⚠️ | FRED key | | **Truncated to about 3 years (starts 2023-11-27)**. Archive it daily. Long-history proxy: `BAA10Y` (D-001) |
| FOMC calendar + statements | federalreserve.gov HTML + `/feeds/press_monetary.xml`, `press_all.xml` | ✅ | none | | No calendar API, so scrape the HTML or hard-code dates once a year |
| Fiscal policy | Congress.gov API | 🔑 | api.data.gov key | 5,000 req/hr | |
| Fundamentals | SEC EDGAR companyfacts (XBRL) | ✅ | User-Agent header | 10 req/s | Use the nightly `companyfacts.zip` bulk file. The User-Agent must be a generic app name plus contact info set in `.env` |
| Earnings calendar | api.nasdaq.com (unofficial) | ⚠️ | none | undocumented | Fallback: Finnhub free tier (🔑, about 1 month ahead) |
| BTC dominance | CoinGecko Demo `/global` | 🔑 | demo key | Plan for 30/min; 10k credits/month | Attribution required |
| Stablecoin supply | DefiLlama `stablecoins.llama.fi` | ✅ | none | undocumented | |
| Funding rates | OKX public, Hyperliquid `/info` | ✅ | none | per venue | Read-only data only (US persons can't trade on either). We build the aggregate series ourselves (D-006) |
| Open interest | Deribit public, Kraken Futures `/derivatives/api/v3/tickers` | ✅ | none | per venue | D-006 |
| BTC ETF flows | SoSoValue `POST /openapi/v2/etf/historicalInflowChart` | ⚠️ | `x-soso-api-key` | unknown | **Whether the key is free is unverified.** If it isn't free, drop ETF flows from v1 (D-007) |
| News (crypto) | Cointelegraph `https://cointelegraph.com/rss`, Decrypt `https://decrypt.co/feed` | ✅ | none | | Check full-text extraction with trafilatura during the build |
| News (economics) | CNBC Markets `https://www.cnbc.com/id/100003114/device/rss/rss.html`, BBC Business `https://feeds.bbci.co.uk/news/business/rss.xml`, NPR Economy `https://feeds.npr.org/1006/rss.xml` | ✅ | none | | Same |
| Historical S&P 500 membership | github.com/fja05680/sp500 (MIT) | ✅ | none | | 1996 onward. Map tickers to CIK because tickers get reused |
| News metadata backfill (optional) | GDELT DOC API | ✅ ⚠️ | none | 1 request per 5 s | URLs only, no text. Many old URLs are dead or paywalled |
| Push alerts (D-016) | ntfy.sh public server (`POST https://ntfy.sh/<topic>`) | ✅ ⚠️ | none (the topic name acts as the secret) | Daily per-IP message cap on the free server, believed to be ~250/day but **unverified**. Confirm during the Phase 1 build | Anyone who knows the topic can read it, so use a long random `NTFY_TOPIC` in `.env`. Self-hosting is an option later |

## ❌ Not free: flagged and NOT used
| Item | Cost | Replacement |
|---|---|---|
| Coinglass API (funding/OI/ETF flows) | from about $273/yr | OKX + Deribit + Kraken Futures + Hyperliquid public endpoints |
| Alpaca Algo Trader Plus (real-time SIP) | $99/mo | Free IEX feed plus SIP data older than 15 minutes |
| Full ICE BofA HY OAS history | ICE licensing | `BAA10Y` proxy plus our own daily archive |
| Reuters (RSS dead), Bloomberg, WSJ, FT, Economist, NYT | paywall | Free lineup above |
| Farside data resellers (Blockworks, Apify) | paid | SoSoValue, or drop. Farside itself blocks scripts behind Cloudflare |
| DefiLlama Pro | ~$300/mo | The free endpoint is enough |

## Geo-blocked from the US (do not use)
Binance (HTTP 451) and Bybit (HTTP 403) return errors from a US IP, including their public data endpoints.
