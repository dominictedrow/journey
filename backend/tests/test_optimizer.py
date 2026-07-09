"""Module-6 optimizer against in-memory SQLite: candidate generation, spec resolution, and
the adopt / keep-active decision (thresholds + risk), all over deterministic fixtures."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.database.models import Ranking, Strategy, Trade, Wallet, WalletMetrics
from app.strategy_optimizer.candidates import candidate_specs, resolve_config
from app.strategy_optimizer.models import StrategySpec
from app.strategy_optimizer.optimizer import optimize

NOW = datetime(2026, 7, 1, 12, 0, 0, tzinfo=UTC)
W1 = "0x" + "11" * 20
W2 = "0x" + "22" * 20


@pytest.fixture(autouse=True)
def _stub_narrate(monkeypatch):
    """Keep the optimizer hermetic: the advisory LLM is off by default (fails open → None), so
    no test reaches out to a real Ollama. Individual tests override this to assert the narrative
    is purely additive."""

    async def _none(*_a, **_k):
        return None

    monkeypatch.setattr("app.strategy_optimizer.optimizer.narrate", _none)
    return monkeypatch


def _settings(**overrides: object) -> Settings:
    base: dict = {
        "optimizer_objective": "total_return",
        "optimizer_top_n_choices": "2",
        "optimizer_lookback_days": 3650,
        "optimizer_initial_capital": Decimal("1000"),
        "optimizer_min_improvement": Decimal("0"),
        "optimizer_max_drawdown": Decimal("1"),
        "optimizer_min_total_return": Decimal("0"),
        "risk_max_open_positions": 10,
        "risk_max_position_size_pct": Decimal("0.1"),  # 10% of the 1000 initial_capital = 100
    }
    base.update(overrides)
    return Settings(**base)


async def _seed_profitable(session: AsyncSession) -> None:
    """Two ranked wallets that each buy low and sell high inside the lookback window."""
    session.add_all(
        [
            Wallet(address=W1, is_tracked=True),
            Wallet(address=W2, is_tracked=True),
            WalletMetrics(wallet_address=W1, computed_at=NOW, volatility=Decimal("0.2")),
            WalletMetrics(wallet_address=W2, computed_at=NOW, volatility=Decimal("0.4")),
        ]
    )
    session.add_all(
        [
            Ranking(
                computed_at=NOW, category="overall", rank=1, wallet_address=W1, score=Decimal("0.9")
            ),
            Ranking(
                computed_at=NOW, category="overall", rank=2, wallet_address=W2, score=Decimal("0.8")
            ),
        ]
    )
    for i, wallet in enumerate((W1, W2)):
        session.add_all(
            [
                Trade(
                    tx_hash="0x" + f"{i}1".zfill(2) * 32,
                    log_index=0,
                    wallet_address=wallet,
                    token_id=f"tok{i}",
                    side="BUY",
                    price=Decimal("0.40"),
                    size=Decimal("100"),
                    usdc_amount=Decimal("40"),
                    block_number=10 + i,
                    timestamp=NOW - timedelta(days=30),
                ),
                Trade(
                    tx_hash="0x" + f"{i}2".zfill(2) * 32,
                    log_index=0,
                    wallet_address=wallet,
                    token_id=f"tok{i}",
                    side="SELL",
                    price=Decimal("0.60"),
                    size=Decimal("100"),
                    usdc_amount=Decimal("60"),
                    block_number=20 + i,
                    timestamp=NOW - timedelta(days=1),
                ),
            ]
        )
    await session.commit()


def test_candidate_grid_clamped_to_risk_limits() -> None:
    settings = _settings(optimizer_top_n_choices="5,10,50", risk_max_open_positions=10)
    specs = candidate_specs(settings)
    top_ns = sorted({s.top_n for s in specs})
    assert top_ns == [5, 10]  # 50 clamped down to 10, then de-duped with the existing 10
    # top_ns (2) × weightings (3) × sizing modes (3) = 18 candidates, all within limits.
    assert len(specs) == 18
    assert all(s.max_open_positions == 10 and s.max_position_size == Decimal("100") for s in specs)


async def test_resolve_config_selects_topn_and_normalizes_weights(session: AsyncSession) -> None:
    await _seed_profitable(session)
    spec = StrategySpec(
        category="overall",
        top_n=2,
        weighting="score",
        sizing_mode="fixed",
        sizing_value=Decimal("50"),
    )
    config = await resolve_config(
        session, spec, initial_capital=Decimal("1000"), lookback_days=3650, now=NOW
    )
    assert config is not None
    assert config.wallets == [W1, W2]  # ranked order preserved
    # score weighting normalizes to mean 1: weights sum to the wallet count.
    assert sum(config.weights.values()) == pytest.approx(Decimal("2"))
    assert config.weights[W1] > config.weights[W2]  # higher score ⇒ larger allocation


async def test_resolve_config_none_without_rankings(session: AsyncSession) -> None:
    spec = StrategySpec(category="overall", top_n=2)
    config = await resolve_config(
        session, spec, initial_capital=Decimal("1000"), lookback_days=3650, now=NOW
    )
    assert config is None


async def test_optimize_adopts_best_when_no_active(session: AsyncSession) -> None:
    await _seed_profitable(session)
    result = await optimize(session, settings=_settings(), rebuild=False, now=NOW)

    assert result.adopted is True
    assert result.strategy_id is not None
    assert result.best is not None and result.best.objective_value is not None
    assert result.best.objective_value > 0  # profitable window
    assert result.rationale["decision"] == "adopted"
    assert result.rationale["candidates"]  # every evaluated candidate is recorded

    row = await session.get(Strategy, result.strategy_id)
    assert row is not None and row.is_active is True
    assert row.adopted_at is not None  # stamped on adoption (SQLite returns it tz-naive)
    assert row.rationale["summary"].startswith("Adopted")


async def test_optimize_keeps_active_when_below_threshold(session: AsyncSession) -> None:
    await _seed_profitable(session)
    # An active strategy identical to the best candidate: no candidate can beat it by the
    # (deliberately huge) required margin, so nothing is adopted.
    active_spec = StrategySpec(
        category="overall",
        top_n=2,
        weighting="equal",
        sizing_mode="proportional",
        sizing_value=Decimal("1.0"),
        max_open_positions=10,
        max_position_size=Decimal("100"),
    )
    session.add(
        Strategy(
            name="incumbent",
            # Enforceable incumbent: params carry a resolved wallet set. (Spec-only params
            # are the unenforceable case covered by the bootstrap test below.)
            params=active_spec.model_dump(mode="json")
            | {"wallets": [W1, W2], "weights": {W1: "1", W2: "1"}},
            is_active=True,
            created_at=NOW - timedelta(days=7),
        )
    )
    await session.commit()

    result = await optimize(
        session,
        settings=_settings(optimizer_min_improvement=Decimal("10")),
        rebuild=False,
        now=NOW,
    )

    assert result.adopted is False
    assert result.active is not None and result.active.objective_value is not None
    assert result.rationale["decision"] == "kept_active"
    # The incumbent is still the only active strategy.
    incumbent = await session.get(Strategy, 1)
    assert incumbent is not None and incumbent.is_active is True
    recommended = await session.get(Strategy, result.strategy_id)
    assert recommended is not None and recommended.is_active is False


async def test_optimize_replaces_unenforceable_active(session: AsyncSession) -> None:
    """An active strategy without a resolved wallet set (pre-wiring row) cannot be enforced
    by apply_active_strategy — and a same-spec candidate grades identically, so the hurdle
    would deadlock adoption forever. The optimizer must replace it regardless of the margin."""
    await _seed_profitable(session)
    active_spec = StrategySpec(
        category="overall",
        top_n=2,
        weighting="equal",
        sizing_mode="proportional",
        sizing_value=Decimal("1.0"),
        max_open_positions=10,
        max_position_size=Decimal("100"),
    )
    session.add(
        Strategy(
            name="incumbent (spec only)",
            params=active_spec.model_dump(mode="json"),  # no "wallets": unenforceable
            is_active=True,
            created_at=NOW - timedelta(days=7),
        )
    )
    await session.commit()

    result = await optimize(
        session,
        settings=_settings(optimizer_min_improvement=Decimal("10")),
        rebuild=False,
        now=NOW,
    )

    assert result.adopted is True
    assert "no resolved wallets" in result.rationale["summary"]
    adopted = await session.get(Strategy, result.strategy_id)
    assert adopted is not None and adopted.is_active is True
    assert adopted.params.get("wallets")  # the replacement is enforceable
    incumbent = await session.get(Strategy, 1)
    assert incumbent is not None and incumbent.is_active is False


async def test_optimize_no_candidate_when_all_lose(session: AsyncSession) -> None:
    # Ranked wallets but a losing window (buy high, sell low) ⇒ no candidate clears the floor.
    session.add_all([Wallet(address=W1, is_tracked=True)])
    session.add(
        Ranking(
            computed_at=NOW, category="overall", rank=1, wallet_address=W1, score=Decimal("0.9")
        )
    )
    session.add_all(
        [
            Trade(
                tx_hash="0x" + "a1" * 32,
                log_index=0,
                wallet_address=W1,
                token_id="tokA",
                side="BUY",
                price=Decimal("0.60"),
                size=Decimal("100"),
                usdc_amount=Decimal("60"),
                block_number=1,
                timestamp=NOW - timedelta(days=30),
            ),
            Trade(
                tx_hash="0x" + "a2" * 32,
                log_index=0,
                wallet_address=W1,
                token_id="tokA",
                side="SELL",
                price=Decimal("0.40"),
                size=Decimal("100"),
                usdc_amount=Decimal("40"),
                block_number=2,
                timestamp=NOW - timedelta(days=1),
            ),
        ]
    )
    await session.commit()

    result = await optimize(
        session,
        settings=_settings(optimizer_min_total_return=Decimal("0")),
        rebuild=False,
        now=NOW,
    )
    assert result.adopted is False
    assert result.best is None
    assert result.rationale["decision"] == "no_candidate"


async def test_narrative_is_purely_additive(_stub_narrate) -> None:
    """The adoption decision must be byte-for-byte identical whether the advisory LLM answers or
    fails — the only difference in the persisted rationale is the `ai_narrative` field. Run over
    two independent in-memory DBs seeded identically so the decision inputs are the same."""
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

    import app.database.models  # noqa: F401 — register tables
    from app.database.base import Base

    async def _run_once(narrate_stub) -> dict:
        _stub_narrate.setattr("app.strategy_optimizer.optimizer.narrate", narrate_stub)
        engine = create_async_engine("sqlite+aiosqlite://")
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        async with async_sessionmaker(engine, expire_on_commit=False)() as s:
            await _seed_profitable(s)
            result = await optimize(s, settings=_settings(), rebuild=False, now=NOW)
            row = await s.get(Strategy, result.strategy_id)
            rationale = dict(row.rationale)
        await engine.dispose()
        return rationale

    async def _fails(*_a, **_k):
        return None

    async def _succeeds(*_a, **_k):
        return "the model liked it"

    failed = await _run_once(_fails)
    succeeded = await _run_once(_succeeds)

    assert failed["ai_narrative"] is None
    assert succeeded["ai_narrative"] == "the model liked it"
    # Every field except the narrative is identical.
    assert {k: v for k, v in failed.items() if k != "ai_narrative"} == {
        k: v for k, v in succeeded.items() if k != "ai_narrative"
    }
