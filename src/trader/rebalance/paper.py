"""Paper rebalancing on Alpaca (D-032): the D-031 planner, with orders sent to PAPER only.

One run:
1. Account checks: paper account number (``PA...``), ACTIVE, not blocked, cash >= 0,
   no open orders left over, long positions only. Any failure -> NO TRADE.
2. Holdings come from the paper account; prices from Alpaca's IEX snapshots. The D-031
   planner applies the same rules and halts (stale/missing/suspicious price, etc.).
3. Market closed -> the plan is shown, nothing is sent.
4. Sells go first as market/day orders; buys are sent only after every sell has filled,
   as notional (dollar) orders so they can't overspend on a stale price.
5. Every run is appended to ``reports/paper/runs.jsonl``; the peak equity for the drawdown
   alert is kept in ``reports/paper/state.json``.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import time
from dataclasses import asdict
from pathlib import Path
from zoneinfo import ZoneInfo

from trader.rebalance.config import Holdings, RebalanceConfig
from trader.rebalance.planner import Plan, Quote, Trade, plan_rebalance

ET = ZoneInfo("America/New_York")
TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day"}


def _f(x) -> float:
    return float(x) if x not in (None, "") else math.nan


def _floor(x: float, dp: int) -> float:
    return math.floor(x * 10**dp) / 10**dp


def account_checks(acct: dict, positions: list[dict], open_orders: list[dict]) -> str | None:
    if not str(acct.get("account_number", "")).startswith("PA"):
        return "account number doesn't start with 'PA': not a paper account"
    if acct.get("status") != "ACTIVE":
        return f"account status is {acct.get('status')!r}, not ACTIVE"
    if acct.get("trading_blocked") or acct.get("account_blocked"):
        return "account is blocked from trading"
    if _f(acct.get("cash")) < -0.01:
        return f"cash is negative (${_f(acct.get('cash')):,.2f}): margin in use"
    if open_orders:
        return f"{len(open_orders)} open order(s) from an earlier run; wait or cancel them"
    shorts = [p["symbol"] for p in positions if p.get("side") != "long" or _f(p["qty"]) < 0]
    if shorts:
        return f"short positions present: {shorts}"
    return None


def holdings_from(acct: dict, positions: list[dict], peak: float | None) -> Holdings:
    shares = {p["symbol"].upper(): _f(p["qty"]) for p in positions}
    return Holdings(max(0.0, _f(acct["cash"])), shares, peak)


def quotes_from(snaps: dict) -> dict[str, Quote]:
    out = {}
    for sym, s in (snaps or {}).items():
        trade, prev = (s or {}).get("latestTrade") or {}, (s or {}).get("prevDailyBar") or {}
        if "p" in trade and "t" in trade:
            date = dt.datetime.fromisoformat(trade["t"][:19]).date()
            out[sym.upper()] = Quote(_f(trade["p"]), date, _f(prev.get("c")) or None)
    return out


def _order(t: Trade, held: float, today: dt.date) -> dict:
    o = {"symbol": t.symbol, "side": t.side.lower(), "type": "market", "time_in_force": "day",
         "client_order_id": f"rb-{today.isoformat()}-{t.symbol}-{t.side.lower()}"}  # fmt: skip
    if t.side == "SELL":
        # Full exit sells the exact quantity held so no dust is left; otherwise round down.
        qty = held if t.shares >= held - 1e-9 else min(_floor(t.shares, 6), held)
        o["qty"] = f"{qty:.9f}".rstrip("0").rstrip(".")
    else:
        o["notional"] = f"{_floor(t.value, 2):.2f}"
    return o


def _wait(client, orders: list[dict], timeout: float, poll: float) -> list[dict]:
    deadline = time.monotonic() + timeout
    while True:
        orders = [client.get_order(o["id"]) for o in orders]
        if all(o["status"] in TERMINAL for o in orders) or time.monotonic() > deadline:
            return orders
        time.sleep(poll)


def run_once(client, cfg: RebalanceConfig, state_dir: Path, now: dt.datetime | None = None,
             fill_timeout: float = 120.0, poll: float = 2.0) -> dict:  # fmt: skip
    """One paper rebalance. Returns a JSON-able record (also appended to runs.jsonl)."""
    now = now or dt.datetime.now(ET)
    today = now.astimezone(ET).date()
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    rec: dict = {"time": now.isoformat(), "sent": [], "messages": []}

    acct, positions, open_orders = client.account(), client.positions(), client.open_orders()
    equity = _f(acct.get("equity"))
    rec["equity"], rec["cash"] = equity, _f(acct.get("cash"))
    reason = account_checks(acct, positions, open_orders)
    if reason:
        plan = Plan(status="HALT", messages=[f"NO TRADE: {reason}"])
    else:
        peak = max(state.get("peak_equity", 0.0), equity) if math.isfinite(equity) else None
        state["peak_equity"] = peak
        holdings = holdings_from(acct, positions, peak)
        symbols = sorted(set(cfg.targets) | {s for s, n in holdings.shares.items() if n > 0})
        quotes = quotes_from(client.snapshots(symbols))
        plan = plan_rebalance(cfg, holdings, quotes, today)
        if plan.status == "OK" and plan.trades:
            if not client.clock().get("is_open"):
                plan.messages.append("Market closed: plan shown, nothing sent")
            else:
                _execute(client, plan, holdings, today, equity, rec, fill_timeout, poll)
    rec["plan"] = asdict(plan)
    state_path.write_text(json.dumps(state, indent=2))
    with (state_dir / "runs.jsonl").open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
    return rec


def _execute(client, plan, holdings, today, equity, rec, fill_timeout, poll) -> None:
    for t in plan.trades:  # belt and braces on top of the planner's own limits
        if not math.isfinite(t.value) or t.value > equity + 1:
            plan.messages.append(f"NO TRADE: {t.side} {t.symbol} ${t.value:,.2f} > equity")
            plan.status = "HALT"
            return
    sells = [t for t in plan.trades if t.side == "SELL"]
    buys = [t for t in plan.trades if t.side == "BUY"]
    for group, label in ((sells, "sell"), (buys, "buy")):
        if not group:
            continue
        sent = [client.submit_order(_order(t, holdings.shares.get(t.symbol, 0.0), today))
                for t in group]  # fmt: skip
        done = _wait(client, sent, fill_timeout, poll)
        rec["sent"] += [{k: o.get(k) for k in ("symbol", "side", "qty", "notional", "status",
                         "filled_qty", "filled_avg_price", "id")} for o in done]  # fmt: skip
        unfilled = [o["symbol"] for o in done if o["status"] != "filled"]
        if unfilled:
            plan.messages.append(f"{label} orders not filled: {unfilled}")
            if label == "sell" and buys:
                plan.messages.append("Buys skipped because sells didn't all fill")
            return


def format_run(rec: dict) -> str:
    p = rec["plan"]
    lines = [f"[{rec['time'][:16]}] paper equity ${rec['equity']:,.2f}  status {p['status']}"]
    for s, w in p.get("weights", {}).items():
        lines.append(f"  {s:<5} {w:>6.1%}  drift {p['drift'][s]:>+6.1%}")
    for o in rec["sent"]:
        amt = f"{o['qty']} sh" if o.get("qty") else f"${o.get('notional')}"
        px = o.get("filled_avg_price")
        lines.append(f"  SENT {o['side'].upper():<4} {o['symbol']:<5} {amt:<14} -> {o['status']}"
                     + (f" @ ${float(px):,.2f}" if px else ""))  # fmt: skip
    lines += [f"  - {m}" for m in p.get("messages", [])]
    return "\n".join(lines)


def next_run_time(clock: dict, minutes_after_open: int = 30) -> dt.datetime:
    """Next session's open + N minutes (if today's has started, that's tomorrow's)."""
    nxt = dt.datetime.fromisoformat(clock["next_open"])
    return nxt + dt.timedelta(minutes=minutes_after_open)
