"""ActionResult: what every harness action and guard returns (PLAN "Harness API"). Portable."""
from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class ActionResult:
    ok: bool
    reason: str = ""
    cost: Optional[int] = None      # $ the action would cost (negative = refund, e.g. a sale)
    value: Any = None               # what the action produced, e.g. the new tid from place_tower

    def __bool__(self):
        return self.ok
