"""CLOB REST reads (no auth): order book snapshots, used later for paper fills."""

import httpx

from app.config import get_settings
from app.ingest.http import throttled_get


async def get_book(token_id: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient(base_url=settings.clob_api_url, timeout=10) as client:
        # Only 1 retry: this sits on the copy hot path, where a fast fallback beats a slow win.
        response = await throttled_get(
            client, "/book", {"token_id": token_id}, upstream="clob", retries=1
        )
        response.raise_for_status()
        return response.json()
