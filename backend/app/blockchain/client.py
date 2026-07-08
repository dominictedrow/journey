"""Polygon RPC access: a small failover pool over the configured provider URLs."""

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from eth_account import Account
from eth_typing import HexStr
from web3 import AsyncHTTPProvider, AsyncWeb3
from web3.middleware import ExtraDataToPOAMiddleware

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRY_ROUNDS = 2  # passes over the whole provider list before giving up
ROUND_BACKOFF_SECONDS = 1.0  # pause between passes — rides out a shared rate-limit blip


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

    async def aclose(self) -> None:
        """Close every provider's cached aiohttp sessions. Callers create a pool per task run
        (15s live tail), so leaving this to GC leaks a session per run ("Unclosed client
        session" in the worker log)."""
        for w3 in self._w3s:
            try:
                await w3.provider.disconnect()
            except Exception:  # noqa: BLE001 — closing must never mask the task's real outcome
                logger.warning("rpc provider close failed", exc_info=True)

    async def __aenter__(self) -> "RpcPool":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def _try(self, op: Callable[[AsyncWeb3], Awaitable[T]]) -> T:
        last_exc: Exception | None = None
        for round_index in range(RETRY_ROUNDS):
            for w3 in self._w3s:
                try:
                    return await op(w3)
                except Exception as exc:  # noqa: BLE001 — any provider failure means "try next"
                    last_exc = exc
                    logger.warning("rpc provider failed (round %d): %s", round_index + 1, exc)
            if round_index + 1 < RETRY_ROUNDS:
                # Every provider failed this pass — most likely a transient/rate-limit blip,
                # so wait before the next pass instead of hammering them back-to-back.
                await asyncio.sleep(ROUND_BACKOFF_SECONDS * (round_index + 1))
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

    async def call(self, *, to: str, data: str) -> bytes:
        """Read-only contract call (e.g. `balanceOf`)."""

        async def op(w3: AsyncWeb3) -> bytes:
            return await w3.eth.call({"to": to, "data": HexStr(data)})

        return await self._try(op)

    async def send_signed_transaction(
        self, *, to: str, data: str, private_key: str, chain_id: int, gas_limit: int
    ) -> str:
        """Sign and broadcast a write call from the key's own address (EOA gas payer + signer).
        Returns the transaction hash (hex, `0x`-prefixed)."""
        account = Account.from_key(private_key)

        async def op(w3: AsyncWeb3) -> str:
            tx = {
                "chainId": chain_id,
                "to": to,
                "data": data,
                "value": 0,
                "gas": gas_limit,
                "gasPrice": await w3.eth.gas_price,
                "nonce": await w3.eth.get_transaction_count(account.address, "pending"),
            }
            signed = account.sign_transaction(tx)
            tx_hash = await w3.eth.send_raw_transaction(signed.raw_transaction)
            return tx_hash.hex()

        return await self._try(op)

    async def wait_for_receipt(self, tx_hash: str, *, timeout: float) -> Any:
        """Block (async) until `tx_hash` confirms; raises on timeout — the caller decides
        whether that means "still pending, check next sweep" or a real failure."""

        async def op(w3: AsyncWeb3) -> Any:
            return await w3.eth.wait_for_transaction_receipt(HexStr(tx_hash), timeout=timeout)

        return await self._try(op)
