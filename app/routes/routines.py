import json
from datetime import datetime, timezone, timedelta

from flask import Blueprint, render_template, redirect, url_for, request

from app.db import fetchall_dict, fetchone_dict, execute_returning
from app.models.market_data import get_latest, get_latest_by_tier, get_multi_series_history, get_multi_pct_change

bp = Blueprint("routines", __name__)

VALID_PERIODS = ("daily", "weekly", "monthly")

DAILY_KEYS = [
    "BTC_USD", "ETH_USD", "EQUITY_SPY", "EQUITY_QQQ",
    "DXY", "GOLD", "VIX", "FEAR_GREED",
]
WEEKLY_KEYS = [
    "T10Y2Y", "BAMLH0A0HYM2", "DGS10", "DFII10",
    "M2SL", "WALCL", "DEXJPUS", "DEXUSAL",
    "VIX", "DXY", "COPPER", "GOLD", "COPPER_GOLD_RATIO", "MOVE",
    "CPI_YOY", "PPI_YOY", "PAYROLLS",
    "COT_SPX_NET", "COT_GOLD_NET", "BTC_ETF_FLOWS_NET",
]


def _with_history(series_list, days=30):
    if not series_list:
        return series_list
    ids = [s["series_id"] for s in series_list]
    histories = get_multi_series_history(ids, 80)
    pct_changes = get_multi_pct_change(ids, days)
    for s in series_list:
        hist = histories.get(s["series_id"], [])
        s["chart_data"] = [{"x": r["recorded_at"].isoformat(), "y": float(r["value"])} for r in hist]
        s["week_pct"] = pct_changes.get(s["series_id"])
    return series_list


def _fetch_by_keys(keys):
    latest = get_latest()
    result = []
    for key in keys:
        s = dict(latest[key]) if key in latest else {
            "series_id": key, "metric_name": key,
            "value": None, "text_value": None, "tier": None, "category": None,
        }
        result.append(s)
    return _with_history(result, 30)


def _fetch_news(hours, limit=30):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    return fetchall_dict("""
        SELECT id, source_name, title, url, published_at, category
        FROM news_items
        WHERE is_active = TRUE AND published_at >= %s
        ORDER BY published_at DESC LIMIT %s
    """, (cutoff, limit))


def _portfolio_summary():
    stock_cash_row  = fetchone_dict("SELECT balance FROM portfolio_cash WHERE asset_type='stock'")
    crypto_cash_row = fetchone_dict("SELECT balance FROM portfolio_cash WHERE asset_type='crypto'")
    cost_rows = fetchall_dict("""
        SELECT asset_type, COALESCE(SUM(cost_basis), 0) AS cost
        FROM portfolio_positions WHERE status='open'
        GROUP BY asset_type
    """)
    count_row = fetchone_dict("SELECT COUNT(*) AS n FROM portfolio_positions WHERE status='open'")
    snap_rows = fetchall_dict("""
        SELECT DISTINCT ON (asset_type) asset_type, positions_value
        FROM portfolio_snapshots ORDER BY asset_type, snapshot_date DESC
    """)

    stock_cash  = float(stock_cash_row["balance"])  if stock_cash_row  else 10000.0
    crypto_cash = float(crypto_cash_row["balance"]) if crypto_cash_row else 10000.0
    costs = {r["asset_type"]: float(r["cost"]) for r in cost_rows}
    snaps = {r["asset_type"]: float(r["positions_value"]) for r in snap_rows if r["positions_value"]}

    stock_cost  = costs.get("stock",  0.0)
    crypto_cost = costs.get("crypto", 0.0)
    stock_pos   = snaps.get("stock",  stock_cost)
    crypto_pos  = snaps.get("crypto", crypto_cost)

    return {
        "stock_cash":             stock_cash,
        "crypto_cash":            crypto_cash,
        "stock_positions_value":  stock_pos,
        "crypto_positions_value": crypto_pos,
        "stock_total":            stock_cash + stock_pos,
        "crypto_total":           crypto_cash + crypto_pos,
        "grand_total":            stock_cash + stock_pos + crypto_cash + crypto_pos,
        "stock_pnl":              stock_pos - stock_cost,
        "crypto_pnl":             crypto_pos - crypto_cost,
        "open_count":             int(count_row["n"]) if count_row else 0,
    }


def _get_routine_data(period):
    if period == "daily":
        return {
            "series": _fetch_by_keys(DAILY_KEYS),
            "news":   _fetch_news(24),
        }

    if period == "weekly":
        sector_rows = fetchall_dict("""
            SELECT DISTINCT ON (series_id)
                series_id, metric_name, value, text_value, tier, category, fetched_at
            FROM market_data WHERE series_id LIKE 'SECTOR_%%'
            ORDER BY series_id, fetched_at DESC
        """)
        _with_history(sector_rows, 7)
        sector_rows.sort(key=lambda r: r.get("metric_name") or r["series_id"])
        return {
            "t1_series":     _fetch_by_keys(WEEKLY_KEYS),
            "sector_series": sector_rows,
            "news":          _fetch_news(168),
        }

    # monthly
    t1 = list(get_latest_by_tier(1))
    t2 = list(get_latest_by_tier(2))
    _with_history(t1, 30)
    _with_history(t2, 30)
    imf_gdp  = [s for s in t2 if s["series_id"].startswith("IMF_GDP_")]
    imf_cpi  = [s for s in t2 if s["series_id"].startswith("IMF_CPI_")]
    other_t2 = [s for s in t2 if not s["series_id"].startswith("IMF_")]

    return {
        "t1_series": t1,
        "imf_gdp":   imf_gdp,
        "imf_cpi":   imf_cpi,
        "other_t2":  other_t2,
        "portfolio": _portfolio_summary(),
        "news":      _fetch_news(720),
    }


@bp.route("/routines/<period>", methods=["GET", "POST"])
def routine(period):
    if period not in VALID_PERIODS:
        return redirect(url_for("home.index"))

    data = _get_routine_data(period)

    if request.method == "POST":
        written = (request.form.get("written_text") or "").strip()
        if written:
            all_series = (data.get("series") or []) + (data.get("t1_series") or [])
            auto_snap = {
                s["series_id"]: {
                    "value": float(s["value"]) if s.get("value") is not None else None,
                    "label": s.get("metric_name") or s["series_id"],
                }
                for s in all_series if s.get("value") is not None
            }
            row = execute_returning("""
                INSERT INTO journal_entries (entry_type, auto_data_snapshot, written_text)
                VALUES (%s, %s, %s) RETURNING id
            """, (period, json.dumps(auto_snap), written))
            return redirect(url_for("journal.entry", entry_id=row[0]))
        return redirect(url_for("home.index"))

    return render_template(f"routines/{period}.html", **data)
