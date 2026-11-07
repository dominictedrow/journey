"""Collect live prices for currently-held portfolio tickers.

The static `prices.py` / `crypto.py` collectors only cover a fixed dashboard universe
(SPY/QQQ/sectors/BTC/ETH/SOL …). Any other ticker bought in the paper portfolio has no
live price, so its P&L would silently sit at $0. This collector reads the open positions,
fetches a price for each held ticker not already refreshed by the static collectors, and
stores it under the SAME series_id the portfolio model looks up
(EQUITY_<ticker> for stocks, <ticker>_USD for crypto) so P&L reflects the market.
"""
import time
import requests
from datetime import datetime, timezone

from app.db import fetchall_dict
from collectors.base import save_market_data

TV_SCAN   = "https://scanner.tradingview.com/{screener}/scan"
TV_SEARCH = "https://symbol-search.tradingview.com/symbol_search/"
TV_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/",
}
_EXCHANGE_SCREENER = {
    "NASDAQ": "america", "NYSE": "america", "AMEX": "america",
    "CBOE": "america", "BATS": "america",
}

CG_SEARCH = "https://api.coingecko.com/api/v3/search"
CG_PRICE  = "https://api.coingecko.com/api/v3/simple/price"
CG_HEADERS = {"User-Agent": "MarketResearchBot/1.0"}


def _series_id(ticker, asset_type):
    """Mirror app.models.portfolio._series_id_for so the portfolio model finds these rows."""
    return f"{ticker}_USD" if asset_type == "crypto" else f"EQUITY_{ticker}"


def _stock_price(ticker):
    """Resolve an arbitrary US stock ticker to a price via TradingView search + scan."""
    sr = requests.get(
        TV_SEARCH,
        params={"text": ticker, "type": "stock", "lang": "en"},
        headers={"User-Agent": TV_HEADERS["User-Agent"],
                 "Origin": TV_HEADERS["Origin"], "Referer": TV_HEADERS["Referer"]},
        timeout=10,
    )
    if sr.status_code != 200 or not sr.json():
        return None
    hit = sr.json()[0]
    exchange  = hit.get("exchange", "NASDAQ")
    screener  = _EXCHANGE_SCREENER.get(exchange, "america")
    tv_symbol = f"{exchange}:{hit.get('symbol', ticker)}"

    pr = requests.post(
        TV_SCAN.format(screener=screener),
        json={"symbols": {"tickers": [tv_symbol], "query": {"types": []}}, "columns": ["close"]},
        headers=TV_HEADERS, timeout=10,
    )
    data = pr.json().get("data", [])
    if data and data[0].get("d") and data[0]["d"][0] is not None:
        return float(data[0]["d"][0])
    return None


def _crypto_price(ticker):
    """Resolve an arbitrary crypto symbol to a USD price via CoinGecko search + price."""
    sr = requests.get(CG_SEARCH, params={"query": ticker}, headers=CG_HEADERS, timeout=15)
    sr.raise_for_status()
    coins = sr.json().get("coins", [])
    if not coins:
        return None
    # Search results are ranked by market cap; prefer an exact symbol match.
    match = next((c for c in coins if c.get("symbol", "").upper() == ticker), coins[0])
    cid = match.get("id")
    if not cid:
        return None
    pr = requests.get(CG_PRICE, params={"ids": cid, "vs_currencies": "usd"},
                      headers=CG_HEADERS, timeout=15)
    pr.raise_for_status()
    return pr.json().get(cid, {}).get("usd")


def _needs_refresh(series_ids):
    """Return the series_ids that lack a row fetched in the last hour, so we skip
    tickers already kept fresh by the static prices.py / crypto.py collectors."""
    if not series_ids:
        return set()
    rows = fetchall_dict("""
        SELECT DISTINCT series_id FROM market_data
        WHERE series_id = ANY(%s) AND fetched_at > NOW() - INTERVAL '1 hour'
    """, (list(series_ids),))
    fresh = {r["series_id"] for r in rows}
    return set(series_ids) - fresh


def collect():
    held = fetchall_dict(
        "SELECT DISTINCT ticker, asset_type FROM portfolio_positions WHERE status = 'open'"
    )
    if not held:
        print("holdings: no open positions")
        return

    wanted = {_series_id(h["ticker"].upper(), h["asset_type"]): h for h in held}
    to_fetch = _needs_refresh(set(wanted.keys()))

    now = datetime.now(timezone.utc)
    rows = []
    for sid in to_fetch:
        h = wanted[sid]
        ticker = h["ticker"].upper()
        atype  = h["asset_type"]
        try:
            price = _crypto_price(ticker) if atype == "crypto" else _stock_price(ticker)
        except Exception as e:
            print(f"holdings: {ticker} ({atype}) error: {e}")
            price = None
        if price is None:
            print(f"holdings: no price for {ticker} ({atype})")
            continue
        rows.append({
            "series_id":   sid,
            "metric_name": ticker,
            "value":       float(price),
            "text_value":  None,
            "source":      "CoinGecko" if atype == "crypto" else "TradingView",
            "category":    "crypto" if atype == "crypto" else "equity",
            "tier":        2 if atype == "crypto" else 3,
            "recorded_at": now,
        })
        time.sleep(1)  # be gentle on the free endpoints

    save_market_data(rows)
    print(f"holdings: saved {len(rows)} held-ticker prices")


if __name__ == "__main__":
    collect()
