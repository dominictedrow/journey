"""Phase 10 — JSON API endpoints."""
import json
import requests
from datetime import datetime, timezone

from flask import Blueprint, jsonify

from app.db import fetchall_dict
from app.models.market_data import get_latest
from config.settings import resource_path

bp = Blueprint("api", __name__)

_SOURCES_PATH = resource_path("config/sources.json")

def _load_intervals():
    with open(_SOURCES_PATH) as f:
        series = json.load(f)["series"]
    return {s["id"]: s["refresh_seconds"] for s in series}

_INTERVALS = _load_intervals()

_TV_SCAN   = "https://scanner.tradingview.com/{screener}/scan"
_TV_SEARCH = "https://symbol-search.tradingview.com/symbol_search/"
_TV_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
    "Origin": "https://www.tradingview.com",
    "Referer": "https://www.tradingview.com/",
}
_EXCHANGE_SCREENER = {
    "NASDAQ": "america", "NYSE": "america", "AMEX": "america",
    "CBOE": "america", "BATS": "america",
    "BINANCE": "crypto", "COINBASE": "crypto", "BITFINEX": "crypto", "BITSTAMP": "crypto",
}


@bp.route("/api/data-freshness")
def data_freshness():
    rows = fetchall_dict("""
        SELECT DISTINCT ON (series_id) series_id, fetched_at
        FROM market_data
        ORDER BY series_id, fetched_at DESC
    """)
    now = datetime.now(timezone.utc)
    result = {}
    for row in rows:
        sid = row["series_id"]
        fa = row["fetched_at"]
        if fa.tzinfo is None:
            fa = fa.replace(tzinfo=timezone.utc)
        seconds_old = int((now - fa).total_seconds())
        interval = _INTERVALS.get(sid, 3600)
        if seconds_old > interval * 5:
            status = "red"
        elif seconds_old > interval * 2:
            status = "amber"
        else:
            status = "ok"
        result[sid] = {
            "seconds_since_fetch": seconds_old,
            "refresh_seconds": interval,
            "status": status,
        }
    return jsonify(result)


@bp.route("/api/ticker-price/<ticker>")
def ticker_price(ticker):
    ticker = ticker.upper().strip()
    if not ticker or len(ticker) > 20:
        return jsonify({"error": "invalid ticker"}), 400

    # Check market_data first — covers all 34+ symbols already collected by prices.py
    latest = get_latest()
    for candidate in (f"EQUITY_{ticker}", f"{ticker}_USD", ticker):
        if candidate in latest and latest[candidate]["value"] is not None:
            return jsonify({"ticker": ticker, "price": float(latest[candidate]["value"])})

    # Fall back to TradingView: symbol search to get exchange, then scan for price
    try:
        sr = requests.get(
            _TV_SEARCH,
            params={"text": ticker, "type": "stock", "lang": "en"},
            headers={"User-Agent": _TV_HEADERS["User-Agent"],
                     "Origin": _TV_HEADERS["Origin"],
                     "Referer": _TV_HEADERS["Referer"]},
            timeout=8,
        )
        if sr.status_code != 200 or not sr.json():
            return jsonify({"error": "symbol not found"}), 404
        hit = sr.json()[0]
        exchange  = hit.get("exchange", "NASDAQ")
        screener  = _EXCHANGE_SCREENER.get(exchange, "america")
        tv_symbol = f"{exchange}:{hit.get('symbol', ticker)}"

        pr = requests.post(
            _TV_SCAN.format(screener=screener),
            json={"symbols": {"tickers": [tv_symbol], "query": {"types": []}},
                  "columns": ["close"]},
            headers=_TV_HEADERS,
            timeout=10,
        )
        data = pr.json().get("data", [])
        if data and data[0].get("d") and data[0]["d"][0] is not None:
            return jsonify({"ticker": ticker, "price": float(data[0]["d"][0])})
        return jsonify({"error": "no price data"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 502
