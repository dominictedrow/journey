"""Application settings.

Single source of truth for every external endpoint, credential, and limit (plan.md:
"centralize them all in config.py so a changed endpoint is a one-line fix").

Polymarket endpoints below are the documented defaults; verify against
https://docs.polymarket.com before wiring each integration (plan.md endpoint-accuracy note).
"""

from decimal import Decimal
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- Core services -------------------------------------------------------------------
    database_url: str = "postgresql+asyncpg://copytrader:copytrader@localhost:5432/copytrader"
    redis_url: str = "redis://localhost:6379/0"

    # --- Polymarket / Polygon endpoints ---------------------------------------------------
    polygon_chain_id: int = 137
    polygon_rpc_urls: str = ""  # comma-separated pool (Alchemy/Infura/QuickNode)
    gamma_api_url: str = "https://gamma-api.polymarket.com"
    data_api_url: str = "https://data-api.polymarket.com"
    clob_api_url: str = "https://clob.polymarket.com"
    clob_ws_url: str = "wss://ws-subscriptions-clob.polymarket.com"
    # Trader leaderboard lives on the Data-API host at /v1/leaderboard — verified against
    # docs.polymarket.com/api-reference/core/get-trader-leaderboard-rankings on 2026-07-02.
    leaderboard_url: str = "https://data-api.polymarket.com"

    # --- Outbound API politeness (app/ingest/http.py) ---------------------------------------
    # Retries per GET against the free Polymarket APIs (429/5xx/transport), with exponential
    # backoff from this base. Per-upstream request throttles live in app/ingest/http.py.
    http_retries: int = 4
    http_backoff_seconds: Decimal = Decimal("0.5")

    # --- Wallet discovery (Module 2) --------------------------------------------------------
    discovery_leaderboard_limit: int = 50  # API max is 50 per page
    discovery_holder_markets: int = 20  # top active markets (by volume) scanned for holders

    # --- Autonomous promotion (Module 2 → 3 bridge; discovery.autopromote hourly beat) ------
    # Full-history backfills are the heaviest Data-API reads, so promotion is budgeted:
    # 25/hour ≈ 600 wallets/day flowing into metrics + rankings, well inside the shared
    # throttle. The cap bounds the live tail's maker-topic list and the hourly metrics sweep.
    # Widened from the original 8/300 so enough wallets are tracked to clear the ≥1 trade/day
    # floor without relying solely on the activity governor's runtime overrides (Module D).
    autopromote_per_run: int = 25
    max_tracked_wallets: int = 1000

    # On-chain contract addresses (Polygon) — verified against
    # docs.polymarket.com/resources/contracts on 2026-07-02.
    ctf_exchange_address: str = "0xE111180000d2663C0091e4f400237545B87B996B"
    neg_risk_ctf_exchange_address: str = "0xe2222d279d744050d28e00520010520000310F59"
    conditional_tokens_address: str = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045"
    # Current collateral is pUSD (CollateralToken proxy); field name kept for continuity.
    usdc_address: str = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB"
    # Pre-pUSD exchanges (CTF Exchange, NegRisk CTF Exchange) — wallet history before the
    # collateral migration lives here, so the indexer scans these too.
    legacy_exchange_addresses: str = (
        "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E,0xC5d563A36AE78145C45a50134d48A1215220f80a"
    )

    # --- Indexer tuning ---------------------------------------------------------------------
    chain_confirmations: int = 5  # blocks behind head to avoid reorgs
    # RPC is now used ONLY for the live tail (small, recent ranges). Historical backfill goes
    # through the Data-API (app/ingest/history.py), because free RPC tiers cap eth_getLogs hard:
    # Alchemy free = 10-block range, QuickNode Discover = 5. 10 matches Alchemy's inclusive cap.
    rpc_log_chunk_size: int = 10  # eth_getLogs block-range per live-tail request (Alchemy cap)
    # Unused by the Data-API history path (kept for the legacy RPC backfill_wallet only).
    backfill_start_block: int = 33_605_403  # legacy CTF Exchange deployment (Polygon)

    # --- Execution ------------------------------------------------------------------------
    execution_mode: Literal["paper", "live"] = "paper"
    live_trading_armed: bool = False

    # Copy portfolio (Module 9/10). Sizing mirrors the backtester's semantics:
    # fixed = USDC per copied BUY; percentage = fraction of equity per BUY;
    # proportional = multiple of the source trade's own notional.
    paper_starting_cash: Decimal = Decimal("10000")
    copy_sizing_mode: Literal["fixed", "percentage", "proportional"] = "proportional"
    copy_sizing_value: Decimal = Decimal("1.0")
    # Live copying only fires on fresh fills; the detector ignores source trades older than
    # this (a full backfill would otherwise be replayed as stale copies).
    copy_max_trade_age_seconds: int = 300

    # CLOB credentials + signing key: live execution only, never committed (docs/deploy.md)
    polymarket_api_key: str = ""
    polymarket_api_secret: str = ""
    polymarket_api_passphrase: str = ""
    wallet_private_key: str = ""
    # Polymarket signature scheme: 0 = EOA (the signing key holds funds), 1 = email/magic
    # proxy, 2 = browser (Gnosis Safe) proxy. Non-EOA schemes trade from `clob_funder_address`.
    clob_signature_type: int = 0
    clob_funder_address: str = ""

    # --- Live execution tuning (Phase 8, Module 9 live path) --------------------------------
    # A live copy is a marketable Fill-And-Kill limit: it crosses the book by this fraction of
    # the source price so it fills (fully or partially) immediately, then cancels the rest
    # rather than resting as a stale maker order chasing a source we can no longer track.
    live_marketable_buffer_pct: Decimal = Decimal("1.0")  # cross by 1% of the source price
    live_max_retries: int = 3  # attempts when a placement fails transiently (nonce/network)
    live_retry_backoff_seconds: Decimal = Decimal("0.5")  # linear backoff between attempts

    # --- On-chain redemption (Module E — settled live positions) ----------------------------
    # Neg Risk Adapter (`Polymarket/neg-risk-ctf-adapter`), verified against its on-chain
    # source on 2026-07-08: multi-outcome markets redeem here (`redeemPositions(bytes32
    # conditionId, uint256[] amounts)`), not through the standard CTF contract.
    neg_risk_adapter_address: str = "0xd91E80cF2E7be2e162c6513ceD06f1dD0dA35296"
    redemption_max_retries: int = 3  # attempts when tx submission fails transiently
    redemption_retry_backoff_seconds: Decimal = Decimal("2.0")  # linear backoff between attempts
    redemption_gas_limit: int = 300_000  # generous static limit; redemptions don't loop on-chain
    redemption_receipt_timeout_seconds: int = 120  # wait for confirmation before treating as sent

    # --- Risk limits (Module 8 defaults; USDC unless noted) --------------------------------
    # Every exposure cap is a fraction of current equity, not a flat dollar figure — they all
    # scale with capital automatically instead of binding arbitrarily once the book grows past
    # whatever dollar figure they were sized for. Nested so no single lever can alone reach the
    # portfolio-wide ceiling: position (10%) < per-market (15%) < per-wallet (20%) < total (60%).
    risk_max_total_exposure_pct: Decimal = Decimal("0.60")
    risk_max_daily_loss: Decimal = Decimal("100")
    risk_max_weekly_loss: Decimal = Decimal("250")
    risk_max_monthly_loss: Decimal = Decimal("500")
    risk_max_open_positions: int = 10
    risk_max_position_size_pct: Decimal = Decimal("0.10")
    risk_max_per_market_exposure_pct: Decimal = Decimal("0.15")
    risk_max_per_wallet_exposure_pct: Decimal = Decimal("0.20")
    risk_min_market_liquidity: Decimal = Decimal("10000")
    risk_max_slippage_pct: Decimal = Decimal("2.0")
    risk_max_copy_delay_seconds: int = 60
    # Live mode has no on-chain balance read yet (follow-up once redemption exists), so this
    # is the equity baseline used until the first live snapshot exists: both the live cash
    # reconstruction (portfolio/service.py::current_cash) and the equity fallback the
    # percentage-sizing/exposure-cap math read (portfolio/service.py::current_equity).
    risk_live_starting_equity: Decimal = Decimal("0")

    # --- Strategy optimizer (Module 6; weekly beat) ----------------------------------------
    # The optimizer backtests candidate strategies (Top-N / weighting / sizing grid) over a
    # trailing window and adopts the best only if it beats the active strategy by a margin and
    # clears absolute floors. Objective is the backtest metric candidates are ranked by.
    optimizer_objective: Literal["sharpe", "sortino", "cagr", "total_return"] = "sharpe"
    optimizer_lookback_days: int = 90  # trailing window of source fills each candidate replays
    optimizer_initial_capital: Decimal = Decimal("10000")  # notional for candidate backtests
    # A candidate must beat the active strategy's objective by at least this *relative* margin
    # to be adopted (0.10 = 10% better). With no active strategy the best passing candidate wins.
    optimizer_min_improvement: Decimal = Decimal("0.10")
    # Absolute adoption floors — a candidate that fails either is never adopted, no matter how it
    # ranks: it must be net-positive and keep max drawdown under this cap.
    optimizer_min_total_return: Decimal = Decimal("0")
    optimizer_max_drawdown: Decimal = Decimal("0.5")
    optimizer_top_n_choices: str = "5,10,20"  # selection sizes explored (clamped to max positions)

    # --- Ollama advisory layer (Module G) --------------------------------------------------
    # Narrative-only: the local LLM annotates strategy-adoption and activity-governor decisions
    # with a plain-English rationale. It NEVER blocks or gates a decision and fails open (returns
    # None) on any error. In Docker the backend/worker reach the `ollama` service on the `core`
    # network; the default host resolves there. A small CPU-friendly instruction model keeps the
    # single-operator deployment light — the client pulls it lazily on first use.
    ollama_url: str = "http://ollama:11434"
    ollama_model: str = "llama3.2:3b"

    @property
    def optimizer_top_n_list(self) -> list[int]:
        seen: list[int] = []
        for part in self.optimizer_top_n_choices.split(","):
            part = part.strip()
            if part and int(part) not in seen:
                seen.append(int(part))
        return seen

    @property
    def polygon_rpc_url_list(self) -> list[str]:
        return [u.strip() for u in self.polygon_rpc_urls.split(",") if u.strip()]

    @property
    def exchange_address_list(self) -> list[str]:
        """Every exchange contract whose fills we index (current + legacy generations)."""
        raw = [
            self.ctf_exchange_address,
            self.neg_risk_ctf_exchange_address,
            *self.legacy_exchange_addresses.split(","),
        ]
        return [a.strip() for a in raw if a.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
