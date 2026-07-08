from decimal import Decimal

import pytest

from app.config import Settings


def test_defaults_are_safe() -> None:
    settings = Settings(_env_file=None)
    assert settings.execution_mode == "paper"
    assert settings.live_trading_armed is False
    assert settings.risk_max_total_exposure == Decimal("5000")


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "live")
    monkeypatch.setenv("POLYGON_RPC_URLS", "https://rpc-a.example, https://rpc-b.example")
    monkeypatch.setenv("RISK_MAX_OPEN_POSITIONS", "3")
    settings = Settings(_env_file=None)
    assert settings.execution_mode == "live"
    assert settings.polygon_rpc_url_list == ["https://rpc-a.example", "https://rpc-b.example"]
    assert settings.risk_max_open_positions == 3
