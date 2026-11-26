"""Config loader (D-014).

Base YAML per component (``configs/<stem>.yaml`` -> top-level key ``<stem>``), deep-merged with a
thin profile overlay (``configs/profiles/<profile>.yaml``), validated with pydantic
(``extra="forbid"``). Secrets come only from the environment / ``.env``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from qt.common.paths import CONFIG_DIR, ENV_FILE, visible_files

PROFILES = ("dev", "paper", "backtest")
DEFAULT_PROFILE = "dev"

# Paper trading only (CLAUDE.md hard rule). Not configurable on purpose.
ALPACA_PAPER_BASE_URL = "https://paper-api.alpaca.markets"


class ConfigError(ValueError):
    """Raised for any invalid configuration, so startup fails loudly."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------- db.yaml
class PoolCfg(_Strict):
    min_size: int = Field(ge=0)
    max_size: int = Field(ge=1)
    max_idle_s: int = Field(ge=1)


class HypertableCfg(_Strict):
    chunk_days: int = Field(ge=1)
    compress_after_days: int = Field(ge=1)


class DbCfg(_Strict):
    host: str
    port: int
    backup_port: int
    name: str
    connect_timeout_s: int = Field(ge=1)
    pool: PoolCfg
    hypertables: dict[str, HypertableCfg]


# ---------------------------------------------------------------- app.yaml
class LoggingCfg(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"]
    dir: str
    retention_days: int = Field(ge=1)


class AlertPriorityCfg(_Strict):
    info: int = Field(ge=1, le=5)
    warn: int = Field(ge=1, le=5)
    critical: int = Field(ge=1, le=5)


class AlertsCfg(_Strict):
    enabled: bool
    provider: Literal["ntfy"]
    server: str
    coalesce_window_s: int = Field(ge=0)
    timeout_s: float = Field(gt=0)
    priority: AlertPriorityCfg
    os_fallback: bool


class ReportsCfg(_Strict):
    daily_dir: str


class AppCfg(_Strict):
    logging: LoggingCfg
    alerts: AlertsCfg
    reports: ReportsCfg


# ---------------------------------------------------------------- backup.yaml
class BackupCfg(_Strict):
    evening_time_et: str
    dump_dir: str
    keep_dumps: int = Field(ge=1)

    @field_validator("evening_time_et")
    @classmethod
    def _hhmm(cls, v: str) -> str:
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v):
            raise ValueError("evening_time_et must be HH:MM (24h)")
        return v


# ---------------------------------------------------------------- calendar.yaml
class CalendarCfg(_Strict):
    exchange: str
    tz: str
    start: str
    decision_offset_min: int = Field(ge=0)


# ---------------------------------------------------------------- collect.yaml
_HHMM = re.compile(r"([01]\d|2[0-3]):[0-5]\d(:[0-5]\d)?")


def _check_hhmm(v: str) -> str:
    if not _HHMM.fullmatch(v):
        raise ValueError(f"{v!r} must be HH:MM or HH:MM:SS (24h)")
    return v


class MonthDayCfg(_Strict):
    month: int = Field(ge=1, le=12)
    day: int = Field(ge=1, le=31)


class PitCfg(_Strict):
    yfinance_xnys_offset_min: int = Field(ge=0)
    futures_release_et: str
    fred_release_time_et: dict[str, str]
    # series whose ALFRED vintage date is earlier than FRED actually posts them (D-029)
    fred_release_lag_sessions: dict[str, int] = {}
    edgar_fallback_days: int = Field(ge=0)
    fomc_schedule_known: MonthDayCfg
    congress_backfill_et: str
    earnings_backfill_lead_days: int = Field(ge=0)
    sosovalue_history_et: str

    @field_validator("futures_release_et", "congress_backfill_et", "sosovalue_history_et")
    @classmethod
    def _time(cls, v: str) -> str:
        return _check_hhmm(v)

    @field_validator("fred_release_time_et")
    @classmethod
    def _times(cls, v: dict[str, str]) -> dict[str, str]:
        if "default" not in v:
            raise ValueError("fred_release_time_et needs a 'default' entry")
        for t in v.values():
            _check_hhmm(t)
        return v


class HttpCfg(_Strict):
    timeout_s: float = Field(gt=0)
    retries: int = Field(ge=0)
    backoff_base_s: float = Field(ge=0)
    backoff_max_s: float = Field(ge=0)
    jitter_s: float = Field(ge=0)
    user_agent: str
    per_host_rps: dict[str, float]

    @field_validator("per_host_rps")
    @classmethod
    def _rps(cls, v: dict[str, float]) -> dict[str, float]:
        if "default" not in v:
            raise ValueError("per_host_rps needs a 'default' entry")
        if any(r <= 0 for r in v.values()):
            raise ValueError("per_host_rps values must be > 0")
        return v


class FreshnessRule(_Strict):
    critical: bool
    max_hours: float = Field(gt=0)
    # True for collectors that run once per XNYS session (stock_daily): max_hours then counts
    # from the first session decision time after the last ok run, so weekends and holidays
    # don't read as stale. False: max_hours since the last ok run.
    session_paced: bool = False


class FreshnessCfg(_Strict):
    default: FreshnessRule
    collectors: dict[str, FreshnessRule]

    def rule(self, collector: str) -> FreshnessRule:
        return self.collectors.get(collector, self.default)


class SimhashCfg(_Strict):
    bits: Literal[64]
    shingle: int = Field(ge=1)
    max_hamming: int = Field(ge=0, le=64)
    window_days: int = Field(ge=1)


class NewsCfg(_Strict):
    min_words: int = Field(ge=0)
    simhash: SimhashCfg
    archive_dir: str
    max_new_per_feed: int = Field(ge=1)
    feed_fail_alert_after: int = Field(ge=1)
    feeds: dict[str, str]


class KindRule(_Strict):
    pattern: str
    kind: str

    @field_validator("pattern")
    @classmethod
    def _regex(cls, v: str) -> str:
        re.compile(v)
        return v


class FedCfg(_Strict):
    feeds: dict[str, str]
    kind_rules: list[KindRule]
    feed_default_kind: dict[str, str]
    speaker_roles: dict[str, str]
    default_role: str
    feed_fail_alert_after: int = Field(ge=1)


class WeeklyCfg(_Strict):
    day: Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
    time_et: str

    @field_validator("time_et")
    @classmethod
    def _time(cls, v: str) -> str:
        return _check_hhmm(v)


class UniverseCfg(_Strict):
    index: str
    candidates: int = Field(ge=1)
    membership_start: str
    extra_etfs: list[str]


class AlpacaCfg(_Strict):
    data_url: str
    stock_feed: Literal["sip", "iex"]
    recent_guard_min: int = Field(ge=15)
    page_limit: int = Field(ge=1, le=10000)
    adjustments: dict[str, str] = Field(min_length=1)
    multi_symbols: int = Field(ge=1, le=1000)
    prices_start: str
    crypto_loc: str
    crypto_start: str


class CryptoCfg(_Strict):
    quote: str
    stablecoin_exclude: list[str]


class MembershipCfg(_Strict):
    fja_url: str
    wikipedia_url: str
    wikipedia_table_id: str
    wikipedia_min_rows: int = Field(ge=1)
    gap_sessions: int = Field(ge=1)
    rename_lookback_days: int = Field(ge=1)
    crosscheck_report: str


class EdgarCfg(_Strict):
    tickers_url: str
    max_rps: float = Field(gt=0, le=10)
    facts_url: str
    submissions_url: str
    submissions_file_url: str
    cache_dir: str
    concepts: list[str] = Field(min_length=1)

    @field_validator("concepts")
    @classmethod
    def _concepts(cls, v: list[str]) -> list[str]:
        for c in v:
            if c.count(":") != 1:
                raise ValueError(f"concept {c!r} must be taxonomy:Concept")
        return v


class YfTicker(_Strict):
    rule: Literal["xnys", "futures_fx"]
    asset_class: Literal["etf", "index", "future", "fx"]


class YfinanceCfg(_Strict):
    start: str
    tickers: dict[str, YfTicker] = Field(min_length=1)
    refetch_days: int = Field(ge=1)
    sleep_s: float = Field(ge=0)
    cache_dir: str


class CorpActionsCfg(_Strict):
    start: str
    lookahead_days: int = Field(ge=0)
    refetch_days: int = Field(ge=1)
    symbols_per_request: int = Field(ge=1, le=500)
    page_limit: int = Field(ge=1, le=1000)


class StatusRule(_Strict):
    status: str
    type: str | None = None
    pattern: str

    @field_validator("pattern")
    @classmethod
    def _regex(cls, v: str) -> str:
        re.compile(v)
        return v


class CongressCfg(_Strict):
    base_url: str
    first_congress: int = Field(ge=93)
    committees: list[str] = Field(min_length=1)
    page_limit: int = Field(ge=1, le=250)
    backfill_budget: int = Field(ge=1)
    status_rules: list[StatusRule]
    event_statuses: list[str]
    tags: dict[str, str]

    @field_validator("tags")
    @classmethod
    def _regexes(cls, v: dict[str, str]) -> dict[str, str]:
        for p in v.values():
            re.compile(p)
        return v


class EarningsCfg(_Strict):
    url: str
    user_agent: str
    lookahead_days: int = Field(ge=1)
    backfill_start: str
    backfill_budget: int = Field(ge=1)


class CoingeckoCfg(_Strict):
    base_url: str
    history_days: int = Field(ge=1)
    refetch_days: int = Field(ge=1)
    ids: dict[str, str] = Field(min_length=1)


class SosovalueCfg(_Strict):
    base_url: str
    country_code: str
    symbols: list[str] = Field(min_length=1)
    history_limit: int = Field(ge=1, le=300)
    daily_limit: int = Field(ge=1, le=300)


class DefillamaCfg(_Strict):
    stablecoins_url: str


class OkxCfg(_Strict):
    base_url: str
    inst: dict[str, str]
    default_interval_h: float = Field(gt=0)
    page_limit: int = Field(ge=1, le=100)


class HyperliquidCfg(_Strict):
    url: str
    interval_h: float = Field(gt=0)


class DeribitInst(_Strict):
    name: str
    oi_unit: Literal["usd", "base"]


class DeribitCfg(_Strict):
    url: str
    inst: dict[str, DeribitInst]


class KrakenCfg(_Strict):
    url: str
    inst: dict[str, str]


class DerivsCfg(_Strict):
    symbols: list[str] = Field(min_length=1)
    funding_start: str
    okx: OkxCfg
    hyperliquid: HyperliquidCfg
    deribit: DeribitCfg
    kraken: KrakenCfg


class FredCfg(_Strict):
    base_url: str
    series: list[str] = Field(min_length=1)
    realtime_origin: str
    max_vintages_per_request: int = Field(ge=1)
    page_limit: int = Field(ge=1)
    archive_series: list[str] = []


class ScheduleCfg(_Strict):
    news_every_min: int = Field(ge=1)
    crypto_utc: str
    congress_et: str
    earnings_et: str
    replan_et: str
    weekly: WeeklyCfg
    summary_et: str
    heartbeat_min: int = Field(ge=1)
    heartbeat_gap_min: int = Field(ge=1)
    freshness_check_min: int = Field(ge=1)
    misfire_grace_s: int = Field(ge=1)
    run_dir: str

    @field_validator("crypto_utc", "congress_et", "earnings_et", "replan_et", "summary_et")
    @classmethod
    def _time(cls, v: str) -> str:
        return _check_hhmm(v)


class SuccessorCfg(_Strict):
    symbol: str
    start: date
    end: date


class IdentityOverrideCfg(_Strict):
    security_key: str
    listed_until: date | None = None
    exclude_sources: list[str] = []
    successor: SuccessorCfg | None = None
    note: str


class IdentityCheckCfg(_Strict):
    sources: list[str]
    min_median_dollar_volume: float = Field(gt=0)
    min_bars: int = Field(ge=1)


class CollectCfg(_Strict):
    pit: PitCfg
    http: HttpCfg
    freshness: FreshnessCfg
    news: NewsCfg
    fed: FedCfg
    universe: UniverseCfg
    alpaca: AlpacaCfg
    crypto: CryptoCfg
    membership: MembershipCfg
    edgar: EdgarCfg
    yfinance: YfinanceCfg
    corp_actions: CorpActionsCfg
    congress: CongressCfg
    earnings: EarningsCfg
    coingecko: CoingeckoCfg
    defillama: DefillamaCfg
    sosovalue: SosovalueCfg
    derivs: DerivsCfg
    fred: FredCfg
    schedule: ScheduleCfg
    identity_overrides: list[IdentityOverrideCfg] = []
    identity_check: IdentityCheckCfg

    def override(self, security_key: str | None) -> IdentityOverrideCfg | None:
        for o in self.identity_overrides:
            if o.security_key == security_key:
                return o
        return None


# ---------------------------------------------------------------- aliases.yaml
class ManualAlias(_Strict):
    ticker: str
    alias: str
    kind: Literal["manual", "short", "ticker"] = "manual"
    case_sensitive: bool = False
    needs_cue: bool = False
    valid_from: str | None = None
    valid_to: str | None = None


class AliasesCfg(_Strict):
    finance_cues: list[str]
    strip_suffixes: list[str]
    strip_parentheticals: bool = True
    min_short_len: int = Field(default=3, ge=1)
    manual: list[ManualAlias]


# ---------------------------------------------------------------- features.yaml (Phase 3)
class FeatureBarsCfg(_Strict):
    source_priority: list[str] = Field(min_length=1)
    yfinance_classes: list[str]
    drop_zero_volume_classes: list[str] = []


class ManualActionCfg(_Strict):
    security_key: str
    kind: str
    ex_date: date
    rate: float | None = None
    old_rate: float | None = None
    new_rate: float | None = None
    special: bool = False
    source: str


class AdjustCfg(_Strict):
    splits: bool = True
    dividends: bool = True
    special_dividends: bool = True
    ignore_kinds: list[str] = []
    max_dividend_frac: float = Field(default=0.9, gt=0, lt=1)
    split_check_tol: float = Field(default=0.693, gt=0)
    split_bar_window_days: int = Field(default=10, ge=1)
    manual: list[ManualActionCfg] = []


class FeaturesCfg(_Strict):
    bars: FeatureBarsCfg
    adjust: AdjustCfg


class Config(_Strict):
    db: DbCfg
    app: AppCfg
    backup: BackupCfg
    calendar: CalendarCfg
    collect: CollectCfg
    aliases: AliasesCfg
    features: FeaturesCfg


# ---------------------------------------------------------------- secrets
class Secrets(_Strict):
    postgres_user: str = "qt"
    postgres_password: SecretStr | None = None
    database_url: SecretStr | None = None
    backup_database_url: SecretStr | None = None
    ntfy_topic: SecretStr | None = None
    alpaca_api_key_id: SecretStr | None = None
    alpaca_api_secret_key: SecretStr | None = None
    fred_api_key: SecretStr | None = None
    congress_api_key: SecretStr | None = None
    coingecko_demo_api_key: SecretStr | None = None
    sosovalue_api_key: SecretStr | None = None
    finnhub_api_key: SecretStr | None = None
    sec_edgar_user_agent: SecretStr | None = None  # contains Dom's contact email: keep secret

    def has(self, name: str) -> bool:
        """True if the secret field ``name`` is set to a non-empty value."""
        v = getattr(self, name)
        if v is None:
            return False
        return bool(v.get_secret_value() if isinstance(v, SecretStr) else v)

    @field_validator("alpaca_api_key_id")
    @classmethod
    def _paper_key(cls, v: SecretStr | None) -> SecretStr | None:
        if v is not None and v.get_secret_value() and not v.get_secret_value().startswith("PK"):
            raise ValueError("ALPACA_API_KEY_ID must be a PAPER key (starts with 'PK')")
        return v


def _secrets_from_env() -> Secrets:
    def get(name: str) -> str | None:
        val = os.environ.get(name)
        return val if val else None

    raw = {
        "postgres_user": get("POSTGRES_USER") or "qt",
        "postgres_password": get("POSTGRES_PASSWORD"),
        "database_url": get("DATABASE_URL"),
        "backup_database_url": get("BACKUP_DATABASE_URL"),
        "ntfy_topic": get("NTFY_TOPIC"),
        "alpaca_api_key_id": get("ALPACA_API_KEY_ID"),
        "alpaca_api_secret_key": get("ALPACA_API_SECRET_KEY"),
        "fred_api_key": get("FRED_API_KEY"),
        "congress_api_key": get("CONGRESS_API_KEY"),
        "coingecko_demo_api_key": get("COINGECKO_DEMO_API_KEY"),
        "sosovalue_api_key": get("SOSOVALUE_API_KEY"),
        "finnhub_api_key": get("FINNHUB_API_KEY"),
        "sec_edgar_user_agent": get("SEC_EDGAR_USER_AGENT"),
    }
    try:
        return Secrets(**raw)
    except Exception as e:  # pydantic ValidationError -> ConfigError, without echoing values
        raise ConfigError(f"invalid secrets in environment: {_errors(e)}") from None


# ---------------------------------------------------------------- loader
@dataclass(frozen=True)
class Settings:
    profile: str
    config: Config
    secrets: Secrets
    config_hash: str


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overlay`` onto ``base``. Overlay keys must already exist in base."""
    out = dict(base)
    for k, v in overlay.items():
        if k not in base:
            raise ConfigError(f"profile key '{k}' does not exist in the base config")
        if isinstance(v, dict) and isinstance(base[k], dict):
            out[k] = deep_merge(base[k], v)
        else:
            out[k] = v
    return out


def _read_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a mapping")
    return data


def config_hash(resolved: dict[str, Any]) -> str:
    canonical = json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _errors(e: Exception) -> str:
    errs = getattr(e, "errors", None)
    if callable(errs):
        return "; ".join(f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in errs())
    return str(e)


def load(
    profile: str | None = None,
    config_dir: Path | None = None,
    env_file: Path | None = None,
) -> Settings:
    config_dir = config_dir or CONFIG_DIR
    env_file = env_file or ENV_FILE
    if env_file.exists():
        load_dotenv(env_file, override=False)

    profile = profile or os.environ.get("QT_PROFILE") or DEFAULT_PROFILE
    if profile not in PROFILES:
        raise ConfigError(f"unknown profile '{profile}', expected one of {PROFILES}")

    base: dict[str, Any] = {p.stem: _read_yaml(p) for p in visible_files(config_dir, "*.yaml")}
    overlay_path = config_dir / "profiles" / f"{profile}.yaml"
    overlay = _read_yaml(overlay_path) if overlay_path.exists() else {}
    resolved = deep_merge(base, overlay)

    # Env port overrides keep docker-compose and the app pointing at the same ports.
    db = resolved.get("db", {})
    for env_name, key in (("QT_DB_PORT", "port"), ("QT_DB_BACKUP_PORT", "backup_port")):
        if os.environ.get(env_name) and isinstance(db, dict):
            db[key] = int(os.environ[env_name])

    try:
        cfg = Config.model_validate(resolved)
    except Exception as e:
        raise ConfigError(f"invalid config (profile={profile}): {_errors(e)}") from None

    return Settings(
        profile=profile,
        config=cfg,
        secrets=_secrets_from_env(),
        config_hash=config_hash({"profile": profile, **cfg.model_dump()}),
    )
