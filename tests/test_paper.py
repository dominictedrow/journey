"""Alpaca paper rebalancer (D-032). A fake client; nothing touches the network."""

import datetime as dt
import json

import pytest

from trader.broker.alpaca import (
    PAPER_URL,
    AlpacaError,
    AlpacaPaper,
    LiveEndpointRefused,
    check_paper_only,
    load_env,
)
from trader.rebalance.config import RebalanceConfig, Rules
from trader.rebalance.paper import ET, next_run_time, run_once

NOW = dt.datetime(2026, 11, 4, 10, 30, tzinfo=ET)
CFG = RebalanceConfig({"SPY": 0.6, "IEF": 0.4}, Rules())
GOOD_ENV = {"ALPACA_API_KEY_ID": "PKTEST", "ALPACA_API_SECRET_KEY": "s"}


class FakeAlpaca:
    def __init__(self, cash=50_000.0, positions=None, prices=None, is_open=True,
                 account_number="PA123", fill="filled", open_orders=None):  # fmt: skip
        self.cash, self.pos = cash, dict(positions or {})
        self.prices = prices or {"SPY": 500.0, "IEF": 100.0}
        self.is_open, self.acct_no, self.fill = is_open, account_number, fill
        self._open = open_orders or []
        self.sent: list[dict] = []
        self.orders: dict[str, dict] = {}

    def account(self):
        eq = self.cash + sum(q * self.prices[s] for s, q in self.pos.items())
        return {"account_number": self.acct_no, "status": "ACTIVE", "cash": str(self.cash),
                "equity": str(eq), "trading_blocked": False, "account_blocked": False}  # fmt: skip

    def positions(self):
        return [{"symbol": s, "qty": str(q), "side": "long"} for s, q in self.pos.items()]

    def open_orders(self):
        return self._open

    def clock(self):
        return {"is_open": self.is_open, "next_open": "2026-11-05T09:30:00-05:00"}

    def snapshots(self, symbols):
        t = "2026-11-04T15:00:00.1Z"
        px = {s: self.prices[s] for s in symbols if s in self.prices}
        return {
            s: {"latestTrade": {"p": p, "t": t}, "prevDailyBar": {"c": p}} for s, p in px.items()
        }

    def submit_order(self, o):
        self.sent.append(o)
        oid = f"id{len(self.sent)}"
        px = self.prices[o["symbol"]]
        qty = float(o["qty"]) if "qty" in o else float(o["notional"]) / px
        if self.fill == "filled":
            sign = 1 if o["side"] == "buy" else -1
            self.pos[o["symbol"]] = self.pos.get(o["symbol"], 0.0) + sign * qty
            self.cash -= sign * qty * px
            if self.pos[o["symbol"]] <= 1e-12:
                del self.pos[o["symbol"]]
        self.orders[oid] = {**o, "id": oid, "status": self.fill, "filled_avg_price": str(px)}
        return self.orders[oid]

    def get_order(self, oid):
        return self.orders[oid]


def run(client, tmp_path, **kw):
    return run_once(client, CFG, tmp_path, now=NOW, fill_timeout=0, poll=0, **kw)


# --- guardrails ---------------------------------------------------------------------------


def test_live_key_refused():
    with pytest.raises(LiveEndpointRefused):
        check_paper_only({"ALPACA_API_KEY_ID": "AKLIVE", "ALPACA_API_SECRET_KEY": "s"})


@pytest.mark.parametrize("var", ["ALPACA_BASE_URL", "APCA_API_BASE_URL"])
def test_live_base_url_refused(var):
    with pytest.raises(LiveEndpointRefused):
        check_paper_only(GOOD_ENV | {var: "https://api.alpaca.markets"})


def test_paper_base_url_and_paper_key_accepted():
    assert check_paper_only(GOOD_ENV | {"ALPACA_BASE_URL": PAPER_URL + "/"}) == ("PKTEST", "s")


def test_missing_keys_rejected():
    with pytest.raises(AlpacaError):
        check_paper_only({})


def test_client_refuses_non_paper_urls():
    c = AlpacaPaper("PKTEST", "s")
    with pytest.raises(LiveEndpointRefused):
        c._req("GET", "https://api.alpaca.markets/v2/account")


def test_load_env_reads_file(tmp_path, monkeypatch):
    for k in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY", "APCA_API_KEY_ID"):
        monkeypatch.delenv(k, raising=False)
    (tmp_path / ".env").write_text('# c\nALPACA_API_KEY_ID="PKX"\nALPACA_API_SECRET_KEY=y\n')
    env = load_env(tmp_path / ".env")
    assert env["ALPACA_API_KEY_ID"] == "PKX" and env["ALPACA_API_SECRET_KEY"] == "y"


def test_non_paper_account_sends_nothing(tmp_path):
    c = FakeAlpaca(account_number="123LIVE")
    rec = run(c, tmp_path)
    assert rec["plan"]["status"] == "HALT" and c.sent == []


def test_open_orders_halt(tmp_path):
    c = FakeAlpaca(open_orders=[{"id": "x"}])
    assert run(c, tmp_path)["plan"]["status"] == "HALT" and c.sent == []


def test_negative_cash_halts(tmp_path):
    c = FakeAlpaca(cash=-500.0, positions={"SPY": 60, "IEF": 200})
    assert run(c, tmp_path)["plan"]["status"] == "HALT" and c.sent == []


# --- behaviour ----------------------------------------------------------------------------


def test_all_cash_buys_to_target_with_notional_orders(tmp_path):
    c = FakeAlpaca()
    rec = run(c, tmp_path)
    assert {o["symbol"]: o["notional"] for o in c.sent} == {"IEF": "20000.00", "SPY": "30000.00"}
    assert all(o["type"] == "market" and o["time_in_force"] == "day" for o in c.sent)
    assert all(o["status"] == "filled" for o in rec["sent"])
    assert c.pos["SPY"] * 500 == pytest.approx(30_000) and c.cash == pytest.approx(0)


def test_market_closed_sends_nothing(tmp_path):
    c = FakeAlpaca(is_open=False)
    rec = run(c, tmp_path)
    assert c.sent == [] and any("Market closed" in m for m in rec["plan"]["messages"])


def test_within_band_sends_nothing(tmp_path):
    c = FakeAlpaca(cash=0.0, positions={"SPY": 60, "IEF": 200})
    assert run(c, tmp_path)["plan"]["status"] == "NO_TRADE_NEEDED" and c.sent == []


def test_sells_before_buys_and_full_exit_is_exact(tmp_path):
    c = FakeAlpaca(cash=0.0, positions={"SPY": 90, "IEF": 50, "VTI": 1.234567891},
                   prices={"SPY": 500.0, "IEF": 100.0, "VTI": 300.0})  # fmt: skip
    run(c, tmp_path)
    sides = [o["side"] for o in c.sent]
    assert sides == sorted(sides, key=lambda s: s != "sell")  # every sell before any buy
    vti = next(o for o in c.sent if o["symbol"] == "VTI")
    assert vti["side"] == "sell" and vti["qty"] == "1.234567891" and "VTI" not in c.pos


def test_unfilled_sells_skip_buys(tmp_path):
    c = FakeAlpaca(cash=0.0, positions={"SPY": 90, "IEF": 50}, fill="new")
    rec = run(c, tmp_path)
    assert [o["side"] for o in c.sent] == ["sell"]
    assert any("Buys skipped" in m for m in rec["plan"]["messages"])


def test_client_order_ids_are_per_day_symbol_side(tmp_path):
    c = FakeAlpaca()
    run(c, tmp_path)
    assert {o["client_order_id"] for o in c.sent} == {"rb-2026-11-04-IEF-buy",
                                                      "rb-2026-11-04-SPY-buy"}  # fmt: skip


def test_peak_tracked_and_drawdown_alert(tmp_path):
    c = FakeAlpaca(cash=0.0, positions={"SPY": 60, "IEF": 200})
    run(c, tmp_path)
    assert json.loads((tmp_path / "state.json").read_text())["peak_equity"] == 50_000
    c.prices = {"SPY": 300.0, "IEF": 100.0}  # equity 38,000: 24% below the peak
    rec = run(c, tmp_path)
    assert any("ALERT" in m for m in rec["plan"]["messages"])
    assert len((tmp_path / "runs.jsonl").read_text().splitlines()) == 2


def test_next_run_time_is_30_min_after_next_open():
    t = next_run_time({"next_open": "2026-11-05T09:30:00-05:00"})
    assert (t.hour, t.minute) == (10, 0)
