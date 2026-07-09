"""Fail-open async client for the local Ollama advisory model.

`narrate()` is the only public entry point. It is deliberately dumb and defensive: it catches
*every* exception, logs a warning, and returns `None` — a caller can always treat a `None` as
"no narrative available" and carry on. Explicit non-goals (per docs/auto.md item G): this must
never be imported into the copy-detect hot path or the risk gate's `check()`; it only annotates
decisions that have already been made on beat-scheduled tasks.

Model provisioning is lazy: the first call pull-if-missing checks `/api/tags` and, if the
configured model isn't cached yet, pulls it once (guarded so concurrent/repeat calls don't
stampede). That first call may therefore take a while (a model download) and return `None`
until the pull completes — acceptable because narration is never on a latency-critical path.
"""

import asyncio
import logging

import httpx

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

# Generous ceiling for the one-time model download — a 3B model over a slow link. This bounds
# only the lazy pull, never `narrate`'s own generation timeout.
_PULL_TIMEOUT = 900.0

_ensure_lock = asyncio.Lock()
_model_ready = False


async def _ensure_model(settings: Settings) -> None:
    """Pull the configured model once if it isn't cached. Errors propagate to `narrate`, which
    catches them; `_model_ready` is only set True on success, so a failure here just lets the
    next call retry the check."""
    global _model_ready
    if _model_ready:
        return
    async with _ensure_lock:
        if _model_ready:
            return
        base = settings.ollama_url.rstrip("/")
        async with httpx.AsyncClient(base_url=base) as client:
            tags = await client.get("/api/tags", timeout=10.0)
            tags.raise_for_status()
            names = {m.get("name") for m in tags.json().get("models", [])}
            if settings.ollama_model not in names:
                logger.info("ollama: pulling model %s (first use)", settings.ollama_model)
                pull = await client.post(
                    "/api/pull",
                    json={"name": settings.ollama_model, "stream": False},
                    timeout=_PULL_TIMEOUT,
                )
                pull.raise_for_status()
        _model_ready = True


async def narrate(
    prompt: str, timeout: float = 10.0, settings: Settings | None = None
) -> str | None:
    """Return a short LLM narrative for `prompt`, or `None` on any failure. Never raises."""
    settings = settings or get_settings()
    try:
        await _ensure_model(settings)
        base = settings.ollama_url.rstrip("/")
        async with httpx.AsyncClient(base_url=base) as client:
            resp = await client.post(
                "/api/generate",
                json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
                timeout=timeout,
            )
            resp.raise_for_status()
            text = (resp.json().get("response") or "").strip()
            return text or None
    except Exception as exc:  # fail open — advisory only, never blocks a decision
        logger.warning("ollama narrate failed (advisory, ignored): %s", exc)
        return None
