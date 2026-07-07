# Current Version — RPC + Historical Ingestion Restructure

_Dated 2026-07-05._ This document describes the changes that moved on-chain ingestion from an
RPC-only design to an **Alchemy + Data-API hybrid**. It complements
[`architecture.md`](architecture.md) (which tracks the overall build).

## Why this change

The historical backfill and live tail were both built on chunked `eth_getLogs` over wide block
ranges (`rpc_log_chunk_size=10_000`). That is incompatible with free RPC tiers, which cap
`eth_getLogs` to a handful of blocks:

| Provider              | `eth_getLogs` range cap | Error code |
| --------------------- | ----------------------- | ---------- |
| QuickNode (Discover)  | 5 blocks                | `-32615`   |
| Alchemy (Free)        | 10 blocks               | `-32600`   |

A wide-range backfill is therefore impossible on a free plan. The fix splits responsibilities:

- **Alchemy RPC** — used **only** for the live tail (small, recent ranges).
- **Polymarket Data-API** — used for **all** historical backfill (free, unlimited, no range cap).

## The bulletproof seam (no double-counting)

A transaction is mined in exactly one block, so a `tx_hash` sits entirely on one side of any
block boundary. Coverage is split at a **frozen per-wallet handoff block `H`**:

- Data-API ingests fills with `timestamp <= timestamp(H)` (blocks `<= H`)
- the live tail ingests blocks `> H`

Polygon enforces strictly-increasing block timestamps, so the split is exact — no overlap, no
gap. The two sources therefore touch **disjoint `tx_hash` sets** and can never collide on the
`(tx_hash, log_index)` unique constraint, even though Data-API rows carry no real log index.
Within a transaction, Data-API rows get a **deterministic synthetic `log_index`** (stable sort of
that tx's fills), so re-running a backfill is perfectly idempotent.

The handoff is frozen write-once in `indexer_state` under stream `history:{address}`, so repeat
backfills never widen the Data-API window into range the live tail already owns.

## Files changed

### Configuration

- **`.env`**
  - `POLYGON_RPC_URLS` → Alchemy Polygon **mainnet** (HTTP; the client uses `AsyncHTTPProvider`,
    not WSS). QuickNode removed. Amoy testnet URL kept only as a commented reference.
  - `RPC_LOG_CHUNK_SIZE=10` — Alchemy's inclusive `eth_getLogs` cap.
  - Populated the four on-chain contract addresses (empty values override the config defaults to
    `""`, which would drop them from `exchange_address_list`).
  - `BACKFILL_START_BLOCK` left in place but documented as unused by the Data-API path.
- **`backend/app/config.py`**
  - `rpc_log_chunk_size` default `10_000` → `10`, with comments explaining it is now live-tail-only.

### New ingestion path

- **`backend/app/ingest/history.py`** (new)
  - `backfill_wallet_history(address)` — Data-API backfill up to the frozen handoff block, then
    rebuild positions. Idempotent; returns rows inserted.
  - `_resolve_handoff(...)` — derives and freezes the per-wallet handoff block (live checkpoint for
    tracked wallets, current head for analyse-only wallets).
  - `data_api_trade_to_row(...)` — maps a Data-API `/trades` record to a `trades` row
    (`source='data_api'`, `block_number=None`).
  - `_with_synthetic_log_index(...)` — assigns each fill a deterministic per-tx ordinal.
- **`backend/app/ingest/data_api.py`**
  - `get_trades` now walks **descending `end=<timestamp>` time-windows** to fetch complete history.
    Polymarket caps offset paging at 3000; windowing reaches unlimited depth. Windows are
    non-overlapping (`end = oldest - 1`), so no fill is fetched twice.
  - `_page_window` replaces `_paged`: stops gracefully at the offset cap (HTTP 400) instead of
    crashing, so an unattended backfill never dies on a high-volume wallet.
  - `PAGE_SIZE` `500` → `1000`; added `MAX_OFFSET = 3000`.

### Schema

- **`backend/app/database/models/trade.py`** — added `source` column (`'chain'` | `'data_api'`,
  server default `'chain'`) for ingestion provenance / auditability.
- **`backend/app/database/models/indexer_state.py`** — `stream` widened `String(32)` → `String(80)`
  (per-wallet stream names like `history:0x<40>` overflowed the old length).
- **Migrations (new):**
  - `d3b8f1c4a9e2_trade_source_column.py` — adds `trades.source`.
  - `e4c9a2f7b6d1_widen_indexer_state_stream.py` — widens `indexer_state.stream` to 80.

### Bug fixes surfaced during wiring

These were latent — the affected code had never run because RPC was never configured before.

- **`backend/app/blockchain/client.py`** — `RpcPool` now injects `ExtraDataToPOAMiddleware`.
  Polygon is a POA chain; without it, `get_block` fails on the oversized `extraData` header (this
  would also have broken the live tail's timestamp lookups).
- **`backend/app/blockchain/normalize.py`**
  - `insert_trades` now **batches at 1000 rows** — a single 42k-row insert exceeded Postgres's
    32767 bind-parameter limit.
  - `compute_positions` now orders by `(timestamp, block_number or 0, log_index)` instead of
    `(block_number, log_index)`, so Data-API rows (null `block_number`) sequence correctly.
    Backward-compatible: on-chain-only data is unaffected because block timestamps are monotonic.
  - `fill_to_trade` now tags on-chain rows with `source='chain'`.

### Task wiring

- **`backend/app/workers/tasks/analytics.py`** and **`backend/app/workers/tasks/ingest.py`** —
  `analytics.backfill_and_compute` and `chain.backfill_wallet` now call `backfill_wallet_history`
  (Data-API) instead of the RPC `backfill_wallet`. The legacy RPC `backfill_wallet` remains in
  `indexer.py` but is no longer wired (free RPC tiers cannot run it).

## Verification

Run against the live stack on 2026-07-05:

- **Full-history fetch:** `get_trades` returned **42,036 records** for one wallet via windowed
  pagination (~22s), spanning the wallet's entire timestamp range.
- **Idempotent backfill:** first run inserted 42,036 rows; re-run inserted **0**.
- **No collisions:** 42,036 rows → **42,036 distinct `(tx_hash, log_index)`**; 11,020 positions
  rebuilt; handoff frozen at block 89,703,018; all rows `source='data_api'` with null
  `block_number`.
- **Live tail:** exercised Alchemy `eth_getLogs` in 10-block chunks + POA `get_block`; advanced the
  live checkpoint correctly.
- **Test suite:** all **114 tests pass**.

## Known limitation

For a wallet with a large burst of trades in a **single second** at the offset-cap boundary, a few
of those same-second fills beyond offset 3000 may be missed. Negligible in practice; noted because
the copier is intended to run unattended.

## Deploy

The Data-API + Alchemy config takes effect on service (re)creation. Bring the hybrid live with the
dev override (mounts source + the updated `.env`):

```
docker compose -f docker/docker-compose.yml -f docker/docker-compose.dev.yml up -d
```

Both new migrations are applied automatically by the `api` container's start command
(`alembic upgrade head`).
