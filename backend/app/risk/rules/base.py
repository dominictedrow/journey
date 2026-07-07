"""Rule protocol and shared helpers.

Every rule exposes `name` and a pure `evaluate(ctx, intent, size) -> RuleResult`. Rules that
only constrain risk-*increasing* actions return `allow` for SELLs (exits reduce risk and are
never blocked by exposure/timing caps — only the kill switch stops them).
"""

from decimal import Decimal
from typing import Protocol

from app.risk.models import RiskContext, RuleResult, TradeIntent

QUANTUM = Decimal("0.000001")


def _dec(value: Decimal) -> Decimal:
    return value.quantize(QUANTUM)


class Rule(Protocol):
    name: str

    def evaluate(self, ctx: RiskContext, intent: TradeIntent, size: Decimal) -> RuleResult: ...


def cap_by_notional(size: Decimal, price: Decimal, remaining: Decimal, reason: str) -> RuleResult:
    """Keep `size * price` within `remaining` USDC of headroom.

    Denies when there is no headroom, shrinks the order when it overflows, else allows.
    """
    if price <= 0:
        return RuleResult.allow(size)
    if remaining <= 0:
        return RuleResult.deny(reason)
    max_size = _dec(remaining / price)
    if max_size <= 0:
        return RuleResult.deny(reason)
    if size > max_size:
        return RuleResult.adjust(max_size, reason)
    return RuleResult.allow(size)
