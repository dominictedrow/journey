import json

from flask import Blueprint, redirect, render_template, request, url_for

from app.db import get_conn
from app.models.portfolio import (
    get_cash_balance,
    get_open_positions,
    get_snapshot_history,
    get_trade_history,
    take_daily_snapshot,
)

bp = Blueprint("portfolio", __name__)

VALID_TYPES = ("stocks", "crypto")
# URL param → DB asset_type value
_DB_TYPE = {"stocks": "stock", "crypto": "crypto"}


def _build_context(asset_type):
    db_type  = _DB_TYPE[asset_type]
    positions = get_open_positions(db_type)
    cash      = get_cash_balance(db_type)
    snapshots = get_snapshot_history(db_type, limit=365)
    trades    = get_trade_history(db_type, limit=60)

    positions_value = sum(p['current_value'] for p in positions)
    total_value     = cash + positions_value
    cost_basis_sum  = sum(float(p['cost_basis']) for p in positions)
    total_pnl       = positions_value - cost_basis_sum

    # Snapshot chart data (for JS time-range filter)
    snap_chart = [
        {"x": str(s['snapshot_date']), "y": float(s['total_value'])}
        for s in snapshots
        if s['total_value'] is not None
    ]

    # Doughnut allocation: {label: value}
    donut_labels = [p['ticker'] for p in positions] + ["Cash"]
    donut_values = [p['current_value'] for p in positions] + [cash]

    return {
        "asset_type":       asset_type,
        "db_type":          db_type,
        "positions":        positions,
        "cash":             cash,
        "total_value":      total_value,
        "positions_value":  positions_value,
        "total_pnl":        total_pnl,
        "snap_chart":       snap_chart,
        "donut_labels":     donut_labels,
        "donut_values":     donut_values,
        "trades":           trades,
    }


@bp.route("/portfolio/")
def index():
    return redirect(url_for("portfolio.view", asset_type="stocks"))


@bp.route("/portfolio/<asset_type>")
def view(asset_type):
    if asset_type not in VALID_TYPES:
        return redirect(url_for("portfolio.view", asset_type="stocks"))
    ctx = _build_context(asset_type)
    # Take a snapshot each time the page is loaded (idempotent upsert for today)
    try:
        take_daily_snapshot(_DB_TYPE[asset_type])
    except Exception:
        pass
    return render_template("portfolio/stocks.html", **ctx)


@bp.route("/portfolio/buy", methods=["POST"])
def buy():
    ticker     = (request.form.get("ticker") or "").strip().upper()
    asset_type = request.form.get("asset_type", "stocks")
    if asset_type not in VALID_TYPES:
        asset_type = "stocks"
    db_type = _DB_TYPE[asset_type]

    try:
        entry_price = float(request.form.get("entry_price") or 0)
        quantity    = float(request.form.get("quantity") or 0)
    except ValueError:
        return redirect(url_for("portfolio.view", asset_type=asset_type))

    if not ticker or entry_price <= 0 or quantity <= 0:
        return redirect(url_for("portfolio.view", asset_type=asset_type))

    cost = entry_price * quantity

    # All-or-nothing: read cash, insert position + trade, debit cash, and write the
    # journal entry in one transaction so a crash can't leave cash and positions out of sync.
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT balance FROM portfolio_cash WHERE asset_type = %s FOR UPDATE",
                (db_type,)
            )
            row = cur.fetchone()
            cash = float(row[0]) if row else 0.0
            if cash < cost:
                return redirect(url_for("portfolio.view", asset_type=asset_type))

            cur.execute("""
                INSERT INTO portfolio_positions
                    (asset_type, ticker, entry_price, quantity, cost_basis, entry_date)
                VALUES (%s, %s, %s, %s, %s, CURRENT_DATE)
                RETURNING id
            """, (db_type, ticker, entry_price, quantity, cost))
            position_id = cur.fetchone()[0]

            cur.execute("""
                INSERT INTO trade_history (position_id, trade_type, price, quantity, trade_value)
                VALUES (%s, 'buy', %s, %s, %s)
            """, (position_id, entry_price, quantity, cost))

            cur.execute(
                "UPDATE portfolio_cash SET balance = balance - %s WHERE asset_type = %s",
                (cost, db_type)
            )

            cur.execute("""
                INSERT INTO journal_entries
                    (entry_type, auto_data_snapshot, portfolio_entries_snapshot, written_text)
                VALUES ('portfolio', %s, %s, %s)
                RETURNING id
            """, (
                json.dumps({}),
                json.dumps([{
                    "ticker": ticker, "entry_price": entry_price,
                    "quantity": quantity, "asset_type": db_type,
                }]),
                f"Bought {quantity:g} {ticker} @ ${entry_price:,.4f}",
            ))
            journal_id = cur.fetchone()[0]

            cur.execute(
                "UPDATE portfolio_positions SET journal_entry_id = %s WHERE id = %s",
                (journal_id, position_id)
            )

    return redirect(url_for("portfolio.view", asset_type=asset_type))


@bp.route("/portfolio/sell", methods=["POST"])
def sell():
    try:
        position_id = int(request.form.get("position_id") or 0)
        exit_price  = float(request.form.get("exit_price") or 0)
    except (ValueError, TypeError):
        return redirect(url_for("portfolio.view", asset_type="stocks"))

    if position_id <= 0 or exit_price <= 0:
        return redirect(url_for("portfolio.view", asset_type="stocks"))

    # All-or-nothing: re-read the position under lock, record the sale, close it,
    # and credit cash in one transaction so the trade can't partially apply.
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT asset_type, ticker, entry_price, quantity, cost_basis
                FROM portfolio_positions
                WHERE id = %s AND status = 'open'
                FOR UPDATE
            """, (position_id,))
            pos = cur.fetchone()
            if not pos:
                return redirect(url_for("portfolio.view", asset_type="stocks"))

            db_type     = pos[0]
            asset_type  = "stocks" if db_type == "stock" else "crypto"
            ticker      = pos[1]
            entry_price = float(pos[2])
            quantity    = float(pos[3])
            cost_basis  = float(pos[4])
            trade_value = exit_price * quantity
            pnl         = (exit_price - entry_price) * quantity
            pnl_pct     = pnl / cost_basis * 100 if cost_basis != 0 else 0.0

            cur.execute("""
                INSERT INTO trade_history
                    (position_id, trade_type, price, quantity, trade_value, pnl, pnl_percent)
                VALUES (%s, 'sell', %s, %s, %s, %s, %s)
            """, (position_id, exit_price, quantity, trade_value, pnl, pnl_pct))

            cur.execute(
                "UPDATE portfolio_positions SET status = 'closed' WHERE id = %s",
                (position_id,)
            )

            cur.execute(
                "UPDATE portfolio_cash SET balance = balance + %s WHERE asset_type = %s",
                (trade_value, db_type)
            )

            cur.execute("""
                INSERT INTO journal_entries
                    (entry_type, portfolio_entries_snapshot, written_text)
                VALUES ('portfolio', %s, %s)
            """, (
                json.dumps([{
                    "ticker": ticker, "exit_price": exit_price,
                    "quantity": quantity, "pnl": pnl, "pnl_pct": pnl_pct,
                    "asset_type": db_type,
                }]),
                f"Sold {quantity:g} {ticker} @ ${exit_price:,.4f} | P&L: ${pnl:+,.2f} ({pnl_pct:+.1f}%)",
            ))

    return redirect(url_for("portfolio.view", asset_type=asset_type))
