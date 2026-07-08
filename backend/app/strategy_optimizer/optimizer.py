"""Module 6 — strategy optimizer.

`optimize` is the weekly (beat-scheduled) job: refresh scoring, backtest a grid of candidate
strategies through the *same* Phase-4 engine the UI uses, and adopt the best only when it
clears the risk constraints AND beats the currently-active strategy's objective by the
configured margin. Every run records a machine- and human-readable `rationale`
(Transparency principle); adoption flips `is_active` and stamps `adopted_at`.

Determinism: candidates are resolved against the latest ranking generation and replayed over
a fixed trailing window, so the same DB state yields the same decision. Ties in the objective
break on the strategy's description for a stable winner.
"""

from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.autonomy.overrides import get_override
from app.backtesting.engine import load_backtest_trades, run_backtest
from app.config import Settings, get_settings
from app.database.models import Strategy
from app.ranking.rankings import rebuild_rankings
from app.strategy_optimizer.candidates import candidate_specs, resolve_config
from app.strategy_optimizer.models import CandidateResult, OptimizeResult, StrategySpec


def _objective(metrics: dict, key: str) -> float | None:
    """The metric candidates are ranked by; None when unavailable (disqualifies from ranking)."""
    value = metrics.get(key)
    return float(value) if value is not None else None


def _passes_risk(spec: StrategySpec, metrics: dict, settings: Settings) -> bool:
    """Risk constraints + sanity floors a candidate must clear to be adoptable at all."""
    drawdown = metrics.get("max_drawdown")
    total_return = metrics.get("total_return")
    if drawdown is None or Decimal(str(drawdown)) > settings.optimizer_max_drawdown:
        return False
    if total_return is None or Decimal(str(total_return)) < settings.optimizer_min_total_return:
        return False
    if spec.top_n > settings.risk_max_open_positions:
        return False
    max_position_cap = settings.optimizer_initial_capital * settings.risk_max_position_size_pct
    if spec.max_position_size is not None and spec.max_position_size > max_position_cap:
        return False
    return True


async def _evaluate(
    session: AsyncSession,
    spec: StrategySpec,
    settings: Settings,
    now: datetime,
) -> CandidateResult | None:
    """Resolve, replay, and grade one spec; None when it can't be resolved (no ranked wallets)."""
    config = await resolve_config(
        session,
        spec,
        initial_capital=settings.optimizer_initial_capital,
        lookback_days=settings.optimizer_lookback_days,
        now=now,
    )
    if config is None:
        return None
    result = run_backtest(config, await load_backtest_trades(session, config))
    return CandidateResult(
        spec=spec,
        config=config,
        metrics=result.metrics,
        objective_value=_objective(result.metrics, settings.optimizer_objective),
        passes_risk=_passes_risk(spec, result.metrics, settings),
        wallets=len(config.wallets),
    )


async def _evaluate_active(
    session: AsyncSession, settings: Settings, now: datetime
) -> tuple[Strategy | None, CandidateResult | None]:
    """The current active strategy re-backtested on the same window (the bar to beat)."""
    active = await session.scalar(select(Strategy).where(Strategy.is_active.is_(True)))
    if active is None:
        return None, None
    try:
        spec = StrategySpec.model_validate(active.params)
    except Exception:  # noqa: BLE001 — a malformed stored spec shouldn't crash the run
        return active, None
    return active, await _evaluate(session, spec, settings, now)


def _hurdle(active_objective: float, min_improvement: Decimal) -> float:
    """Objective a candidate must reach to beat the active strategy by the required margin.

    Positive baselines require a relative lift; a zero/negative baseline requires an absolute
    improvement, so a losing active strategy can still be displaced by a merely-better one."""
    margin = float(min_improvement)
    if active_objective > 0:
        return active_objective * (1 + margin)
    absolute = margin if active_objective == 0 else 0.0
    return active_objective + abs(active_objective) * margin + absolute


def _build_rationale(
    *,
    objective: str,
    settings: Settings,
    best: CandidateResult | None,
    active: CandidateResult | None,
    candidates: list[CandidateResult],
    adopted: bool,
    decision: str,
    summary: str,
) -> dict:
    return {
        "decision": decision,
        "adopted": adopted,
        "objective": objective,
        "min_improvement": float(settings.optimizer_min_improvement),
        "max_drawdown_cap": float(settings.optimizer_max_drawdown),
        "lookback_days": settings.optimizer_lookback_days,
        "summary": summary,
        "chosen": best.summary() if best else None,
        "active": active.summary() if active else None,
        "candidates": [c.summary() for c in candidates],
    }


async def optimize(
    session: AsyncSession,
    *,
    settings: Settings | None = None,
    rebuild: bool = True,
    now: datetime | None = None,
) -> OptimizeResult:
    """Run the optimizer once. Adopts a new strategy only when it beats thresholds + risk."""
    settings = settings or get_settings()
    now = now or datetime.now(UTC)
    if rebuild:
        await rebuild_rankings(session)

    extra_top_n = await get_override(session, "optimizer_top_n_extra", 0)
    candidates: list[CandidateResult] = []
    for spec in candidate_specs(settings, extra_top_n=extra_top_n or None):
        evaluated = await _evaluate(session, spec, settings, now)
        if evaluated is not None:
            candidates.append(evaluated)

    eligible = [c for c in candidates if c.passes_risk and c.objective_value is not None]
    best = max(eligible, key=lambda c: (c.objective_value, c.name)) if eligible else None
    active_row, active = await _evaluate_active(session, settings, now)

    if best is None:
        rationale = _build_rationale(
            objective=settings.optimizer_objective,
            settings=settings,
            best=None,
            active=active,
            candidates=candidates,
            adopted=False,
            decision="no_candidate",
            summary="No candidate cleared the risk constraints and adoption floors.",
        )
        return OptimizeResult(
            adopted=False,
            objective=settings.optimizer_objective,
            best=None,
            active=active,
            candidates=candidates,
            rationale=rationale,
        )

    # Decide adoption vs the active strategy's bar.
    best_objective = best.objective_value
    assert best_objective is not None  # eligibility filter guarantees a ranked objective
    if active is None or active.objective_value is None:
        adopted = True
        summary = (
            f"Adopted {best.name}: {settings.optimizer_objective} "
            f"{best_objective:.4f} (no active strategy to beat)."
        )
    elif active_row is None or not (active_row.params or {}).get("wallets"):
        # Pre-wiring active rows carry only the spec, no resolved wallet set — so
        # `apply_active_strategy` cannot enforce them and nothing ever gets copy-enabled.
        # An unenforceable incumbent must not hold the hurdle (it grades identically to a
        # same-spec candidate, deadlocking adoption forever); replace it with the best
        # candidate so the loop gains a concrete selection.
        adopted = True
        summary = (
            f"Adopted {best.name}: {settings.optimizer_objective} {best_objective:.4f} "
            "(active strategy has no resolved wallets to enforce)."
        )
    else:
        hurdle = _hurdle(active.objective_value, settings.optimizer_min_improvement)
        adopted = best_objective >= hurdle
        verb = "Adopted" if adopted else "Kept active strategy"
        summary = (
            f"{verb}: candidate {best.name} {settings.optimizer_objective} "
            f"{best_objective:.4f} vs active {active.objective_value:.4f} "
            f"(hurdle {hurdle:.4f})."
        )

    rationale = _build_rationale(
        objective=settings.optimizer_objective,
        settings=settings,
        best=best,
        active=active,
        candidates=candidates,
        adopted=adopted,
        decision="adopted" if adopted else "kept_active",
        summary=summary,
    )

    # Persist the *resolved* selection alongside the spec: the copy engine and
    # `apply_active_strategy` need the concrete wallet set + weights this adoption was
    # graded on, not a rule that would silently re-resolve against future rankings.
    # StrategySpec ignores the extra keys on re-validation.
    params = best.spec.model_dump(mode="json") | {
        "wallets": list(best.config.wallets),
        "weights": {a: str(w) for a, w in (best.config.weights or {}).items()},
    }
    row = Strategy(
        name=best.name,
        params=params,
        is_active=adopted,
        created_at=now,
        adopted_at=now if adopted else None,
        rationale=rationale,
    )
    if adopted and active_row is not None:
        active_row.is_active = False
    session.add(row)
    await session.commit()

    return OptimizeResult(
        adopted=adopted,
        objective=settings.optimizer_objective,
        strategy_id=row.id,
        best=best,
        active=active,
        candidates=candidates,
        rationale=rationale,
    )
