import time
import requests
from datetime import datetime, timezone
from collectors.base import save_market_data

SCAN_URL = "https://scanner.tradingview.com/{screener}/scan"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/",
}

# (screener, tv_symbol, series_id, label, category, tier)
SYMBOLS = [
    ("america", "CBOE:VIX",     "VIX",         "VIX",                   "volatility", 1),
    ("cfd",     "TVC:DXY",      "DXY",         "DXY Dollar Index",      "fx",         1),
    ("futures", "COMEX:HG1!",   "COPPER",      "Copper Futures",        "commodity",  1),
    ("futures", "COMEX:GC1!",   "GOLD",        "Gold Futures",          "commodity",  1),
    ("america", "ICE:MOVE",     "MOVE",        "MOVE Index (Bond VIX)", "volatility", 1),
    ("futures", "NYMEX:CL1!",   "OIL_WTI",    "WTI Crude Oil",         "commodity",  1),
    ("america", "AMEX:SPY",     "EQUITY_SPY",  "S&P 500 (SPY)",         "equity",     3),
    ("america", "NASDAQ:QQQ",   "EQUITY_QQQ",  "Nasdaq 100 (QQQ)",      "equity",     3),
    ("america", "AMEX:IWM",     "EQUITY_IWM",  "Russell 2000 (IWM)",    "equity",     3),
    ("america", "AMEX:DIA",     "EQUITY_DIA",  "Dow Jones (DIA)",       "equity",     3),
    ("america", "AMEX:XLK",     "SECTOR_XLK",  "Tech (XLK)",            "equity",     3),
    ("america", "AMEX:XLF",     "SECTOR_XLF",  "Financials (XLF)",      "equity",     3),
    ("america", "AMEX:XLE",     "SECTOR_XLE",  "Energy (XLE)",          "equity",     3),
    ("america", "AMEX:XLV",     "SECTOR_XLV",  "Healthcare (XLV)",      "equity",     3),
    ("america", "AMEX:XLI",     "SECTOR_XLI",  "Industrials (XLI)",     "equity",     3),
    ("america", "AMEX:XLP",     "SECTOR_XLP",  "Staples (XLP)",         "equity",     3),
    ("america", "AMEX:XLU",     "SECTOR_XLU",  "Utilities (XLU)",       "equity",     3),
    ("america", "AMEX:XLY",     "SECTOR_XLY",  "Discretionary (XLY)",   "equity",     3),
    ("america", "AMEX:XLB",     "SECTOR_XLB",  "Materials (XLB)",       "equity",     3),
    ("america", "AMEX:XLRE",    "SECTOR_XLRE", "Real Estate (XLRE)",    "equity",     3),
    ("america", "AMEX:XLC",     "SECTOR_XLC",  "Comm Services (XLC)",   "equity",     3),
    ("america", "NASDAQ:TLT",   "BOND_TLT",    "20Y Treasury (TLT)",    "bond",       3),
    ("america", "NASDAQ:IEF",   "BOND_IEF",    "7-10Y Treasury (IEF)",  "bond",       3),
    ("america", "NASDAQ:SHY",   "BOND_SHY",    "1-3Y Treasury (SHY)",   "bond",       3),
    ("america", "AMEX:EWJ",     "INTL_EWJ",    "Japan (EWJ)",           "equity",     2),
    ("america", "AMEX:EWZ",     "INTL_EWZ",    "Brazil (EWZ)",          "equity",     2),
    ("america", "AMEX:EEM",     "INTL_EEM",    "EM (EEM)",              "equity",     2),
    ("america", "AMEX:VEA",     "INTL_VEA",    "Developed ex-US (VEA)", "equity",     2),
    ("america", "NASDAQ:MCHI",  "INTL_MCHI",   "China (MCHI)",          "equity",     2),
    ("america", "AMEX:EWG",     "INTL_EWG",    "Germany (EWG)",         "equity",     2),
    ("forex",   "FX:EURUSD",    "FX_EURUSD",   "EUR/USD",               "fx",         2),
    ("forex",   "FX:GBPUSD",    "FX_GBPUSD",   "GBP/USD",               "fx",         2),
]


def _scan(screener, tv_symbols, _retries=2, _backoff=5):
    """POST to TradingView screener; returns {tv_symbol: price} dict. Retries on 403."""
    url = SCAN_URL.format(screener=screener)
    payload = {
        "symbols": {"tickers": tv_symbols, "query": {"types": []}},
        "columns": ["close"],
    }
    for attempt in range(_retries + 1):
        try:
            resp = requests.post(url, json=payload, headers=HEADERS, timeout=(5, 15))
            if resp.status_code == 403 and attempt < _retries:
                print(f"prices: TradingView /{screener} 403 — retrying in {_backoff}s (attempt {attempt + 1}/{_retries})")
                time.sleep(_backoff)
                continue
            if resp.status_code != 200:
                print(f"prices: TradingView /{screener} returned {resp.status_code}")
                return {}
            result = {}
            for item in resp.json().get("data", []):
                symbol = item.get("s")
                values = item.get("d", [])
                if symbol and values and values[0] is not None:
                    result[symbol] = float(values[0])
            return result
        except Exception as e:
            if attempt < _retries:
                print(f"prices: TradingView /{screener} error (attempt {attempt + 1}): {e} — retrying")
                time.sleep(_backoff)
            else:
                print(f"prices: TradingView /{screener} error: {e}")
    return {}


def collect():
    groups: dict[str, list[str]] = {}
    for (screener, tv_symbol, *_) in SYMBOLS:
        groups.setdefault(screener, []).append(tv_symbol)

    prices: dict[str, float] = {}
    for screener, tv_symbols in groups.items():
        prices.update(_scan(screener, tv_symbols))
        time.sleep(1)

    now = datetime.now(timezone.utc)
    rows = []
    copper_price = None
    gold_price = None

    for (screener, tv_symbol, series_id, label, category, tier) in SYMBOLS:
        price = prices.get(tv_symbol)
        if price is None:
            print(f"prices: no data for {tv_symbol}")
            continue
        if series_id == "COPPER":
            copper_price = price
        if series_id == "GOLD":
            gold_price = price
        rows.append({
            "series_id":   series_id,
            "metric_name": label,
            "value":       price,
            "text_value":  None,
            "source":      "TradingView",
            "category":    category,
            "tier":        tier,
            "recorded_at": now,
        })

    if copper_price and gold_price and gold_price != 0:
        rows.append({
            "series_id":   "COPPER_GOLD_RATIO",
            "metric_name": "Copper/Gold Ratio",
            "value":       round(copper_price / gold_price, 6),
            "text_value":  None,
            "source":      "TradingView",
            "category":    "commodity",
            "tier":        1,
            "recorded_at": now,
        })

    save_market_data(rows)
    print(f"prices: saved {len(rows)} series")


if __name__ == "__main__":
    collect()
