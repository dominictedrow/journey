from datetime import date

from app.db import fetchall_dict, fetchone_dict, execute, execute_returning
from app.models.market_data import get_latest


# Series-id lookup heuristics for live price from market_data
def _series_id_for(ticker, asset_type):
    t = ticker.upper()
    if asset_type == 'crypto':
        return f"{t}_USD"
    return f"EQUITY_{t}"


def get_cash_balance(asset_type):
    row = fetchone_dict(
        "SELECT balance FROM portfolio_cash WHERE asset_type = %s", (asset_type,)
    )
    return float(row['balance']) if row else 0.0


def compute_pnl(entry_price, current_price, quantity):
    return (current_price - entry_price) * quantity


def get_open_positions(asset_type):
    """Returns open positions with live P&L attached. Falls back to cost-basis price if no market data."""
    rows = fetchall_dict("""
        SELECT id, ticker, entry_price, quantity, cost_basis, entry_date, journal_entry_id
        FROM portfolio_positions
        WHERE status = 'open' AND asset_type = %s
        ORDER BY entry_date DESC, id DESC
    """, (asset_type,))

    latest = get_latest()

    for p in rows:
        series_id = _series_id_for(p['ticker'], asset_type)
        if series_id in latest and latest[series_id]['value'] is not None:
            current_price = float(latest[series_id]['value'])
        else:
            current_price = float(p['entry_price'])

        entry_price = float(p['entry_price'])
        quantity    = float(p['quantity'])
        cost_basis  = float(p['cost_basis'])

        p['current_price']  = current_price
        p['current_value']  = current_price * quantity
        p['pnl']            = compute_pnl(entry_price, current_price, quantity)
        p['pnl_pct']        = p['pnl'] / cost_basis * 100 if cost_basis != 0 else 0.0
        p['has_live_price'] = series_id in latest and latest[series_id]['value'] is not None

    return rows


def get_snapshot_history(asset_type, limit=365):
    return fetchall_dict("""
        SELECT snapshot_date, total_value, cash_balance, positions_value
        FROM portfolio_snapshots
        WHERE asset_type = %s
        ORDER BY snapshot_date ASC
        LIMIT %s
    """, (asset_type, limit))


def get_trade_history(asset_type, limit=50):
    return fetchall_dict("""
        SELECT th.id, th.trade_type, th.price, th.quantity, th.trade_value,
               th.pnl, th.pnl_percent, th.trade_date, pp.ticker
        FROM trade_history th
        JOIN portfolio_positions pp ON th.position_id = pp.id
        WHERE pp.asset_type = %s
        ORDER BY th.trade_date DESC
        LIMIT %s
    """, (asset_type, limit))


def take_daily_snapshot(asset_type):
    """Upsert today's portfolio snapshot. Called by scheduler at midnight."""
    positions = fetchall_dict("""
        SELECT ticker, quantity, cost_basis
        FROM portfolio_positions
        WHERE status = 'open' AND asset_type = %s
    """, (asset_type,))

    cash = get_cash_balance(asset_type)
    latest = get_latest()

    positions_value = 0.0
    for p in positions:
        series_id = _series_id_for(p['ticker'], asset_type)
        if series_id in latest and latest[series_id]['value'] is not None:
            positions_value += float(latest[series_id]['value']) * float(p['quantity'])
        else:
            positions_value += float(p['cost_basis'])

    total = cash + positions_value
    today = date.today()

    execute("""
        INSERT INTO portfolio_snapshots
            (snapshot_date, asset_type, total_value, cash_balance, positions_value)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (snapshot_date, asset_type) DO UPDATE
            SET total_value = EXCLUDED.total_value,
                cash_balance = EXCLUDED.cash_balance,
                positions_value = EXCLUDED.positions_value
    """, (today, asset_type, total, cash, positions_value))
