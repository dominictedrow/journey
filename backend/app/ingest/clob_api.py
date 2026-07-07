"""CLOB REST reads (no auth): order book snapshots, used later for paper fills."""

import httpx

from app.config import get_settings


async def get_book(token_id: str) -> dict:
    settings = get_settings()
    async with httpx.AsyncClient(base_url=settings.clob_api_url, timeout=30) as client:
        response = await client.get("/book", params={"token_id": token_id})
        response.raise_for_status()
        return response.json()
