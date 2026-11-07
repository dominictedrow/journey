from datetime import datetime, timezone, timedelta

from flask import Blueprint, render_template
from app.db import fetchall_dict, fetchone_dict
from app.models.market_data import get_latest, get_series_history, get_multi_series_history, get_multi_pct_change

bp = Blueprint("assets", __name__)

EQUITY_KEYS = ["EQUITY_SPY", "EQUITY_QQQ", "EQUITY_IWM", "EQUITY_DIA"]
MACRO_SIGNAL_KEYS = ["VIX", "T10Y2Y", "BAMLH0A0HYM2", "DGS10", "DFII10", "MOVE"]
CRYPTO_PRICE_KEYS = ["BTC_USD", "ETH_USD", "SOL_USD"]
CRYPTO_SIGNAL_KEYS = [
    "BTC_DOMINANCE", "FEAR_GREED", "STABLECOIN_MCAP",
    "DEFI_TVL", "BTC_ETF_FLOWS_NET", "MVRV_ZSCORE", "ALTCOIN_SEASON",
]
FX_KEYS = ["DXY", "DEXJPUS", "DEXUSAL"]
COMMODITY_KEYS = ["COPPER", "GOLD", "OIL_WTI", "COPPER_GOLD_RATIO"]


# Flow / net-positioning series swing through zero, so a price-style percent
# change is meaningless (e.g. latest 0 vs a negative baseline reads as +100%).
# Show the absolute value for these instead of a bogus percent.
NO_PCT_CHANGE_SERIES = {"BTC_ETF_FLOWS_NET", "COT_SPX_NET", "COT_GOLD_NET"}


def _change_pct(hist):
    if len(hist) < 2:
        return None
    first = float(hist[0]["value"]) if hist[0]["value"] is not None else None
    last  = float(hist[-1]["value"]) if hist[-1]["value"] is not None else None
    if first is None or last is None or first == 0:
        return None
    return (last - first) / abs(first) * 100


def _with_history(rows, limit=80):
    if not rows:
        return rows
    ids = [r["series_id"] for r in rows]
    histories = get_multi_series_history(ids, limit)
    for r in rows:
        hist = histories.get(r["series_id"], [])
        r["chart_data"] = [{"x": h["recorded_at"].isoformat(), "y": float(h["value"])} for h in hist]
        r["change_pct"] = None if r["series_id"] in NO_PCT_CHANGE_SERIES else _change_pct(hist)
    return rows


def _fetch_by_keys(keys, latest):
    result = []
    for key in keys:
        if key in latest:
            result.append(dict(latest[key]))
        else:
            result.append({
                "series_id": key, "metric_name": key,
                "value": None, "text_value": None, "tier": None, "category": None,
            })
    return result


def _fetch_news(categories=None, hours=72, limit=40):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    if isinstance(categories, str):
        categories = [categories]
    if categories:
        rows = fetchall_dict("""
            SELECT id, source_name, title, url, published_at, category
            FROM news_items
            WHERE is_active = TRUE AND published_at >= %s AND category = ANY(%s)
            ORDER BY engagement_score DESC, published_at DESC LIMIT %s
        """, (cutoff, list(categories), limit))
    else:
        rows = fetchall_dict("""
            SELECT id, source_name, title, url, published_at, category
            FROM news_items
            WHERE is_active = TRUE AND published_at >= %s
            ORDER BY engagement_score DESC, published_at DESC LIMIT %s
        """, (cutoff, limit))
    return rows


@bp.route("/assets/us")
def assets_us():
    latest = get_latest()

    equities = _fetch_by_keys(EQUITY_KEYS, latest)
    _with_history(equities, 80)

    sector_rows = fetchall_dict("""
        SELECT DISTINCT ON (series_id)
            series_id, metric_name, value, text_value, tier, category, fetched_at
        FROM market_data WHERE series_id LIKE 'SECTOR_%%'
        ORDER BY series_id, fetched_at DESC
    """)
    sector_ids = [s["series_id"] for s in sector_rows]
    sector_histories = get_multi_series_history(sector_ids, 80)
    sector_pct_1m = get_multi_pct_change(sector_ids, 30)
    sector_pct_1w = get_multi_pct_change(sector_ids, 7)
    for s in sector_rows:
        hist = sector_histories.get(s["series_id"], [])
        s["chart_data"] = [{"x": h["recorded_at"].isoformat(), "y": float(h["value"])} for h in hist]
        s["pct_1m"] = sector_pct_1m.get(s["series_id"])
        s["pct_1w"] = sector_pct_1w.get(s["series_id"])
        s["week_pct"] = s["pct_1m"]
    sector_rows.sort(key=lambda r: r.get("metric_name") or r["series_id"])

    macro = _fetch_by_keys(MACRO_SIGNAL_KEYS, latest)
    _with_history(macro, 80)

    news = _fetch_news(categories=["us", "macro"], hours=72, limit=40)

    return render_template("assets/us.html",
                           equities=equities,
                           sectors=sector_rows,
                           macro=macro,
                           news=news)


@bp.route("/assets/international")
def assets_international():
    latest = get_latest()

    fx = _fetch_by_keys(FX_KEYS, latest)
    _with_history(fx, 120)

    commodities = _fetch_by_keys(COMMODITY_KEYS, latest)
    _with_history(commodities, 80)

    imf_gdp = fetchall_dict("""
        SELECT DISTINCT ON (series_id)
            series_id, metric_name, value, text_value, fetched_at
        FROM market_data WHERE series_id LIKE 'IMF_GDP_%%'
        ORDER BY series_id, fetched_at DESC
    """)
    imf_cpi = fetchall_dict("""
        SELECT DISTINCT ON (series_id)
            series_id, metric_name, value, text_value, fetched_at
        FROM market_data WHERE series_id LIKE 'IMF_CPI_%%'
        ORDER BY series_id, fetched_at DESC
    """)

    gdp_by_country = {r["series_id"].replace("IMF_GDP_", ""): r for r in imf_gdp}
    cpi_by_country = {r["series_id"].replace("IMF_CPI_", ""): r for r in imf_cpi}
    countries = sorted(set(list(gdp_by_country.keys()) + list(cpi_by_country.keys())))
    imf_rows = [
        {"country": c, "gdp": gdp_by_country.get(c), "cpi": cpi_by_country.get(c)}
        for c in countries
    ]

    cot_rows = fetchall_dict("""
        SELECT DISTINCT ON (series_id)
            series_id, metric_name, value, text_value, fetched_at
        FROM market_data WHERE series_id IN ('COT_SPX_NET', 'COT_GOLD_NET')
        ORDER BY series_id, fetched_at DESC
    """)

    news = _fetch_news(categories=["international"], hours=168, limit=40)

    return render_template("assets/international.html",
                           fx=fx,
                           commodities=commodities,
                           imf_rows=imf_rows,
                           cot_rows=cot_rows,
                           news=news)


@bp.route("/assets/crypto")
def assets_crypto():
    latest = get_latest()

    prices = _fetch_by_keys(CRYPTO_PRICE_KEYS, latest)
    _with_history(prices, 200)

    signals = _fetch_by_keys(CRYPTO_SIGNAL_KEYS, latest)
    _with_history(signals, 80)

    sig_dict = {s["series_id"]: s for s in signals}

    # BTC dominance needs 200 pts for the dedicated chart; others reuse signals chart_data
    btc_dom_hist = get_series_history("BTC_DOMINANCE", 200)
    btc_dom_chart = [{"x": h["recorded_at"].isoformat(), "y": float(h["value"])} for h in btc_dom_hist]

    fg_chart   = sig_dict.get("FEAR_GREED", {}).get("chart_data", [])
    etf_chart  = sig_dict.get("BTC_ETF_FLOWS_NET", {}).get("chart_data", [])
    stbl_chart = sig_dict.get("STABLECOIN_MCAP", {}).get("chart_data", [])

    news = _fetch_news(categories=["crypto"], hours=168, limit=40)

    return render_template("assets/crypto.html",
                           prices=prices,
                           signals=signals,
                           btc_dom_chart=btc_dom_chart,
                           fg_chart=fg_chart,
                           etf_chart=etf_chart,
                           stbl_chart=stbl_chart,
                           news=news)
