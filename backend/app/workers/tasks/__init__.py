"""Celery task modules (one per module family) + the shared async runner."""

import asyncio
from collections.abc import Coroutine
from typing import Any

from app.database.base import engine


def run_async(coro: Coroutine[Any, Any, Any]) -> Any:
    """Run a coroutine in a fresh event loop, disposing pooled DB connections after —
    asyncpg connections are bound to the loop they were created on."""

    async def wrapper() -> Any:
        try:
            return await coro
        finally:
            await engine.dispose()

    return asyncio.run(wrapper())
