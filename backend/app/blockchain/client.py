"""Polygon RPC access: a small failover pool over the configured provider URLs."""

from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from web3 import AsyncHTTPProvider, AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware

T = TypeVar("T")


def _make_w3(url: str) -> AsyncWeb3:
    w3 = AsyncWeb3(AsyncHTTPProvider(url))
    # Polygon is POA: block headers carry an oversized extraData field that the default
    # validation rejects. This middleware trims it so get_block/get_logs parse correctly.
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3


class RpcPool:
    """Tries each configured RPC URL in order; raises the last error if all fail."""

    def __init__(self, urls: list[str]) -> None:
        if not urls:
            raise RuntimeError("POLYGON_RPC_URLS is not configured")
        self._w3s = [_make_w3(url) for url in urls]

    async def _try(self, op: Callable[[AsyncWeb3], Awaitable[T]]) -> T:
        last_exc: Exception | None = None
        for w3 in self._w3s:
            try:
                return await op(w3)
            except Exception as exc:  # noqa: BLE001 — any provider failure means "try next"
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    async def get_block_number(self) -> int:
        async def op(w3: AsyncWeb3) -> int:
            return await w3.eth.block_number

        return await self._try(op)

    async def get_block(self, number: int) -> Any:
        async def op(w3: AsyncWeb3) -> Any:
            return await w3.eth.get_block(number)

        return await self._try(op)

    async def get_logs(self, params: dict[str, Any]) -> list[Any]:
        async def op(w3: AsyncWeb3) -> list[Any]:
            return list(await w3.eth.get_logs(params))  # type: ignore[arg-type]

        return await self._try(op)
