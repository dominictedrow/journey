"""Runtime-override store: an override-if-present-else-settings-default layer.

`Settings` is `lru_cache`d and env vars aren't safely mutable across worker/beat/API processes
at runtime, so the activity governor (Module D) can't just flip a config value when it decides
to widen the wallet funnel. It persists the decision here instead; every consumer reads through
`get_override` and falls back to its own static settings default when no row exists yet.
"""

from datetime import UTC, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.database.models import RuntimeOverride


async def get_override(session: AsyncSession, key: str, default: int) -> int:
    row = await session.get(RuntimeOverride, key)
    if row is None:
        return default
    return int(row.value["value"])


async def set_override(
    session: AsyncSession, key: str, value: int, reason: str, *, commit: bool = True
) -> RuntimeOverride:
    now = datetime.now(UTC)
    row = await session.get(RuntimeOverride, key)
    if row is None:
        row = RuntimeOverride(key=key, value={"value": value}, reason=reason, updated_at=now)
        session.add(row)
    else:
        row.value = {"value": value}
        row.reason = reason
        row.updated_at = now
    if commit:
        await session.commit()
    return row
