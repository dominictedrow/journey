from datetime import datetime

from sqlalchemy import DateTime, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database.base import Base, JsonB


class RuntimeOverride(Base):
    """A runtime-mutable settings override, keyed by settings field name.

    `Settings` is `lru_cache`d and env vars aren't safely mutable across worker/beat/API
    processes, so the activity governor (Module D) persists its autonomous widening decisions
    here instead of trying to mutate config directly: callers read this as an
    override-if-present-else-settings-default layer (`app.autonomy.overrides.get_override`).
    """

    __tablename__ = "runtime_overrides"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[dict] = mapped_column(JsonB)
    reason: Mapped[str | None] = mapped_column(String(255))
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
