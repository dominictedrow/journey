# Full-Autonomy Remodel: Polymarket Copytrader

## Context

The bot already runs a nearly-complete autonomous loop on Celery beat: wallet discovery → autopromote → metrics → scoring/rankings → strategy backtesting/optimization → strategy enforcement → copy detection → risk gate → paper/live execution → settlement. This is not a prototype — it's built and tested (`backend/tests/test_autonomy_wiring.py`).

Dom wants to go further: the **only** manual actions left should be (1) the kill switch and (2) closing one open position early. Everything else — wallet selection, sizing, strategy adoption, exposure limits, settlement, redemption — must run with zero human input, running toward **live money**, with enough tracked wallets to trade at least once a day, and never more than 60% of portfolio value in open markets at once. A local Ollama layer should be added as an advisory/narrative-only addition — never a blocking gate — per Dom's confirmed choice.

Investigation (3 parallel explore agents + direct file reads) found the loop is more complete than expected, but surfaced concrete gaps, confirmed with Dom:
- A per-wallet manual "disable copy" toggle whose effect is a **permanent human veto** that the autonomous optimizer will never override (`strategy_optimizer/apply.py:32,55-57`) → **remove entirely** (confirmed).
- Several dashboard buttons duplicate beat-scheduled jobs (sweep, rebuild, optimize, revalue, simulate, refresh-metrics, backfill) → strip from the UI so only kill switch + close remain as operator actions.
- The portfolio exposure cap (`risk_max_total_exposure`) is a flat **$5,000** dollar figure, not a %-of-portfolio basis → must become a dynamic ≤60%-of-equity cap.
- While investigating that cap, found `revalue_portfolio_task` (`workers/tasks/copy.py:56`) is only ever invoked with the default `mode="paper"` — **live mode never gets a portfolio snapshot today**, so live equity is always 0 and the loss-limit auto-kill-switch can never fire live. This must be fixed as part of the exposure-cap work, or the new % cap would block every live trade.
- No mechanism guarantees ≥1 trade/day — needs wider defaults plus a self-widening governor.
- Live-mode settlement only updates internal books; actually redeeming winning CTF tokens on-chain is undocumented as manual-only today (`portfolio/settlement.py:13-15`).
- Orphaned live orders are flagged for a human to reconcile by hand (`copy_trading/recovery.py`) with no dedicated remediation path.
- No AI/LLM code exists anywhere yet — Ollama integration is net-new (confirmed: advisory/narrative only, fail-open).

Confirmed with Dom: target is live trading, the wallet veto is removed outright, Ollama is advisory-only.

## Implementation order

`A → B → C → D → F → E → G`

A/B are low-risk cleanup that unblock D. C (dynamic exposure cap + the live-equity fix it exposes) is a prerequisite for D reasoning safely about how far to widen the wallet pool. F (auto-book orphans) is simpler and safer than E (on-chain redemption), so it goes first among the live-trading gaps. G (Ollama) is purely additive and hooks into artifacts C/D/E's work produces, so it goes last.

---

## A. Remove the per-wallet manual veto

- `backend/app/strategy_optimizer/apply.py` — delete `MANUAL_DISABLE_PREFIX` (line 32) and the veto branch (lines 55-57); every strategy-selected wallet unconditionally gets `is_tracked=True`, `copy_enabled=True`, `disabled_reason=None`. Update the module docstring (lines 12-15).
- `backend/app/api/routes/wallets.py` — delete the `PATCH /wallets/{address}` route (lines 118-137) that mutates `copy_enabled`/`is_tracked`. Its only callers are the frontend toggle buttons removed below; nothing else depends on it. Leave `refresh-metrics` (140-145) and `backfill` (148-155) in place — they're debug tools with no gating effect, not part of the veto.
- `backend/app/schemas/wallet.py` — remove `WalletUpdate` if now unused.
- `frontend/src/app/wallets/page.tsx` and `frontend/src/app/wallets/detail/WalletDetail.tsx` — remove the Copy/Track toggle buttons and their mutation hooks; these pages become read-only.
- `frontend/src/lib/api.ts` — remove `updateWallet`.

**Tests:** `backend/tests/test_autonomy_wiring.py::test_apply_respects_manual_veto` (lines 92-104) must be replaced with a test confirming a previously-rotated-out wallet **is** re-enabled once it's back in the strategy's selection (this already works for non-manual reasons — worth an explicit regression test once manual is gone). Check for any wallet-route PATCH test before deleting the route.

## B. Strip redundant manual "run now" dashboard controls

Keep every backend endpoint (useful for curl-based ops/debugging — not part of normal operation), remove the button from the UI in each case:

- `frontend/src/app/discovery/page.tsx` — "Sweep now" button.
- `frontend/src/app/rankings/page.tsx` — "Rebuild" button.
- `frontend/src/app/optimization/page.tsx` — "Run optimizer" button.
- `frontend/src/app/portfolio/page.tsx` — "Revalue now" button only (leave the Close-position button untouched — that's the retained manual action).
- `frontend/src/app/risk/page.tsx` — "Fire simulated trade" button only (leave the kill-switch toggle untouched — that's the other retained manual action).
- `frontend/src/app/wallets/detail/WalletDetail.tsx` — "Refresh metrics" / "Backfill on-chain" buttons (do this in the same pass as A's toggle removal on this file).

No backend or DB changes in this item.

## C. Dynamic exposure cap (≤60% of portfolio value) + fix the live-equity gap

- `backend/app/config.py:107` — replace `risk_max_total_exposure: Decimal = Decimal("5000")` with `risk_max_total_exposure_pct: Decimal = Decimal("0.60")`.
- `backend/app/risk/models.py` — `RiskLimits` (44-73): replace `max_total_exposure` with `max_total_exposure_pct`, read from the new setting in `from_settings`. `RiskContext` (76-89): add `total_equity: Decimal = Decimal(0)`.
- `backend/app/risk/rules/limits.py` — `PortfolioExposureRule` (75-87): store `self.pct` instead of a frozen `self.cap`; compute `cap = ctx.total_equity * self.pct` inside `evaluate()` each call. The sibling rules (`PositionSizeRule`, `MaxOpenPositionsRule`, `PerMarketExposureRule`, `PerWalletExposureRule`) stay as flat-dollar rails — only the whole-portfolio cap moves to a % basis, matching what was asked for.
  - **Open question for Dom, don't guess a number:** `risk_max_position_size` ($100), `risk_max_per_market_exposure` ($200), `risk_max_per_wallet_exposure` ($300) (`config.py:112-115`) are sized for the current small paper cap. Once the portfolio scales to real capital, these flat dollar sub-caps may bind far below 60% of equity, or be irrelevant. Flag this during implementation rather than picking new numbers now — revisit once live capital size is known.
- `backend/app/risk/engine.py::build_context` (57-99) — fetch current equity the same way `CopyEngine._equity` does (`copy_trading/engine.py:63-72`: latest `PortfolioSnapshot.total_value` for the mode, fallback for paper). Set it on `RiskContext.total_equity`.
- **Live-equity fix (required for the above to mean anything in live mode):** `celery_app.py`'s `revalue-portfolio-60s` entry calls `copy.revalue_portfolio` with no `mode` kwarg, and `revalue_portfolio_task(mode: str = "paper")` (`workers/tasks/copy.py:55-56`) therefore only ever snapshots paper. Add a second beat entry, `revalue-portfolio-live-60s`, passing `kwargs={"mode": "live"}`, cheaply no-op'ing when `execution_mode != "live"` (mirror the guard in `reconcile_live_task`, `workers/tasks/copy.py:27-34`). Without this, live `total_equity` is always 0, permanently blocking live BUYs, and the loss-limit auto-kill-switch can never trigger live (no snapshots to diff in `_period_pnl`).
  - Live starting equity currently has no source of truth (`CopyEngine._equity` falls back to `Decimal(0)` for live, `portfolio/service.py` does the same). Ship a config fallback (`risk_live_starting_equity` setting) for now, but note that deriving live equity from the actual on-chain collateral balance of `clob_funder_address` is the more correct long-term approach — worth a follow-up once redemption (item E) exists, since balance and redemption are related.

**DB migration:** none for C itself (no schema changes, pure config/risk-model wiring).

**Tests:** `backend/tests/test_risk_rules.py` — rewrite every `PortfolioExposureRule` test to pass `total_equity=` on `RiskContext` and assert against `equity * pct`; update the `_limits()` helper's `max_total_exposure` kwarg. `backend/tests/test_config.py` — update the assertion on `risk_max_total_exposure` to the new pct field. Grep `backend/tests/` for any other direct `RiskLimits(`/`RiskContext(total_exposure=` construction before landing (renamed field is a breaking signature change).

## D. Guarantee ≥1 trade/day: wider defaults + a self-widening activity governor

**Widen static defaults** (`backend/app/config.py:42-50`): raise `max_tracked_wallets` (300 → larger) and `autopromote_per_run` (8 → larger); tighten `discovery-sweep-6h` in `celery_app.py` to run more often (e.g. every 1-2h — "constantly searching" implies more than every 6h); tighten `optimize-strategies-weekly` to daily, since `optimizer_min_improvement`'s 10% hurdle (`config.py:128`) already prevents strategy thrashing from more frequent re-evaluation.

**New activity governor** — a scheduled task that checks trailing trade frequency (e.g. filled `CopyOrder` count over 24h) and, if below target, autonomously widens the funnel within hard-coded safe bounds (bump effective autopromote budget / tracked-wallet cap, or bias optimizer top-N selection wider), fully logged, no human step.

- New logic module, e.g. `backend/app/autonomy/activity_governor.py`, with a thin Celery task wrapper following the existing split (business logic separate from the `workers/tasks/*.py` wrapper — see `wallet_discovery/autopromote.py` vs `workers/tasks/discovery.py` for the pattern).
- Runtime-mutable overrides need a durable store, since `Settings` is `lru_cache`d and env vars aren't safely mutable across worker/beat/API processes at runtime. Add a new small table, e.g. `runtime_overrides` (key, value JSONB, updated_at, reason) — don't try to force this into `system_flags`, whose `enabled: Boolean` shape doesn't fit numeric overrides. Read it as an override-if-present-else-settings-default layer in `promotion_budget()` (`wallet_discovery/autopromote.py:32-41`) and wherever `optimizer_top_n_list` is consumed (`strategy_optimizer/candidates.py`).
- New beat entry (e.g. `activity-governor-6h`, every 6h so it reacts faster than a full day of silence).
- Log every autonomous widening as a `RiskEvent` (reuse the existing `context: JSONB` audit-row pattern already used for kill-switch auto-engage and orphan capture — type e.g. `"activity_governor_widened"`).

**DB migration:** new `runtime_overrides` table.

**Tests:** update `test_promotion_budget_clamps_to_cohort_cap` (`test_autonomy_wiring.py:288-297`) for the override path; new `backend/tests/test_activity_governor.py` covering trigger/no-op/bounded-widen-doesn't-runaway; update `test_config.py` for changed defaults.

## F. Auto-book orphaned live orders (do before E)

`backend/app/copy_trading/recovery.py::_capture_orphan` (69-100) currently flags an orphaned CLOB order for manual reconciliation and stops. Change it to also call `apply_fill` (already imported) with a distinguishable sentinel wallet, e.g. `source_wallet="reconciled_orphan"`, so cash/positions stay accurate automatically. `OrderState` only carries `token_id`, not `market_id` — resolve it via `load_token_market_map` (the same helper `settlement.py:25` already uses) before calling `apply_fill`. Keep writing the `RiskEvent` for audit visibility, but change `action_taken` from `"flagged"` to `"auto_booked"`. Update the module docstring, which currently argues against auto-booking.

**Tradeoff to flag directly to Dom, not silently decide:** auto-booking removes the safety property that a human trading manually on the same funded wallet outside the bot would show up as a visible, unbooked orphan. Given the whole point of this remodel is that Dom never trades manually except kill-switch/close, this is very likely fine — but it's worth him confirming explicitly rather than the risk being buried in a UI-invisible code change.

**DB migration:** none. **Tests:** find (or add) the recovery test file; assert an orphan now produces a booked position/cash update, not just a flagged row.

## E. Automate on-chain redemption of settled live positions (do after F)

Today `settlement.py:13-15` explicitly documents this as manual ("do that in the Polymarket UI for now"). For live mode this needs to become automatic:

- New module `backend/app/blockchain/redemption.py` calling the Conditional Tokens contract's `redeemPositions` for standard markets. **Verify during implementation, don't guess:** the exact ABI/argument shape (`collateralToken`, `parentCollectionId`, `conditionId`, `indexSets`), and whether negative-risk markets (`neg_risk_ctf_exchange_address`) redeem through a different contract/function — check `docs.polymarket.com/resources/contracts` directly.
- `backend/app/blockchain/client.py::RpcPool` is currently read-only (`get_block_number`/`get_block`/`get_logs`); it needs a new sign-and-send capability using `wallet_private_key` (`config.py:92`). Keep the existing pool-per-run + `aclose()` pattern (memory: RPC provider sessions must be closed after each pool use) rather than opening a new connection style.
- Retry/failure handling modeled on `LiveExecutor` (`copy_trading/executors/live.py:109-146`: bounded retries with linear backoff, terminal failure logged rather than raised) so a stuck redemption never crashes the settlement beat.
- New columns on `PortfolioPosition` (`database/models/portfolio.py:25-41`): `redemption_status`, `redeemed_at`, `redemption_tx_hash` — for idempotency (never re-redeem) and visibility into failures.
- Recommend a **separate** beat task (e.g. `redeem-settled-positions-10m`) rather than inlining into `settle-resolutions-10m`, so on-chain confirmation latency never blocks the internal-books settlement sweep. Gate on `position.mode == "live"`.
- Confirm the signing wallet holds gas (MATIC/POL) — flag whether a balance check/alert is needed.

**DB migration:** new Alembic revision adding the redemption columns to `portfolio_positions`.

**Tests:** new `backend/tests/test_redemption.py` with a stub Web3 client (mirror how live-executor tests fake the CLOB client), covering idempotency and retry/failure marking. Add a `mode="live"` case to `test_autonomy_wiring.py`'s settlement tests (currently `mode="paper"` only, lines 192-229) asserting redemption is triggered for live but not paper.

## G. Ollama advisory layer (net-new, do last)

Confirmed scope: advisory/narrative only, never blocks or gates a decision, fails open on any error.

- `docker/docker-compose.yml` — new `ollama` service (`image: ollama/ollama`, `restart: unless-stopped`, new named volume for model cache, on the existing `core` network so backend/worker reach it at `http://ollama:11434`).
- Model-pull strategy: rather than a separate init container (fragile healthcheck choreography for one-operator scale), have the backend's fail-open client do a pull-if-missing check lazily on first real call — simpler, and keeps the "must never block startup" property in Python instead of compose orchestration.
- New settings: `ollama_url` (`http://ollama:11434`) and `ollama_model` (recommend a small CPU-friendly instruction model, e.g. `qwen2.5:7b` or `llama3.2:3b`) in `config.py`.
- New module `backend/app/ai_advisor/` — `client.py` (thin async `httpx` wrapper, `httpx` is already a dependency) and a `narrate(prompt, timeout=10.0) -> str | None` function catching every exception and returning `None` on any failure, logging a warning, never raising. Explicit non-goal: never import this into the 15s copy-detect hot path or the risk gate's `check()`.
- **Integration 1 — strategy adoption:** in `strategy_optimizer/optimizer.py`, after `_build_rationale()` builds its dict and before commit, call `narrate()` on the existing rationale fields and merge the result into `rationale["ai_narrative"]` (or `None` if unavailable) — no schema change needed, `Strategy.rationale` is already JSONB.
- **Integration 2 — activity governor:** narrate item D's widening decisions into that `RiskEvent`'s `context["ai_narrative"]`.

**DB migration:** none (writes into existing JSONB columns).

**Tests:** new `backend/tests/test_ai_advisor.py` covering fail-open behavior on timeout/connection-refused/malformed response. Add stubbed-`narrate` tests to `test_optimizer.py` confirming adoption decisions are byte-for-byte identical whether `narrate` succeeds or fails.

---

## Verification

- Backend: `cd backend && pytest` after each item (A–G), especially `test_autonomy_wiring.py`, `test_risk_rules.py`, `test_config.py`, and the new test files this plan adds.
- Docker: `docker compose -f docker/docker-compose.yml up -d --build`, confirm all services (including new `ollama`) come up healthy.
- End-to-end smoke test in paper mode first: confirm the wallet pages no longer show Copy/Track toggles, confirm discovery/rankings/optimization/portfolio/risk pages no longer show "run now"/"revalue"/"simulate" buttons but kill switch and position-close still work, watch a few beat cycles to confirm autopromote → strategy apply → copy detect still flows with the veto removed.
- Confirm the new `revalue-portfolio-live-60s` beat entry actually produces `PortfolioSnapshot(mode="live")` rows once live mode is armed, and that `PortfolioExposureRule` correctly computes a nonzero cap from them.
- Before flipping `EXECUTION_MODE=live`/`LIVE_TRADING_ARMED=true` for real: confirm item E's redemption path against Polymarket's actual contracts on a small position first, and confirm the signing wallet is funded for gas.
