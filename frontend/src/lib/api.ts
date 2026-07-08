import { useSettings } from "@/stores/settings";

export interface HealthResponse {
  status: string;
  version: string;
  execution_mode: string;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const base = useSettings.getState().backendUrl;
  const res = await fetch(`${base}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...init?.headers },
  });
  if (!res.ok) {
    throw new ApiError(res.status, await res.text());
  }
  return (await res.json()) as T;
}

export function getHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>("/health");
}

// Numeric money/metric fields arrive as strings (backend serializes Decimal exactly);
// parse for display with the helpers in lib/format.ts.
export interface WalletMetrics {
  computed_at: string;
  roi_lifetime: string | null;
  realized_pnl: string | null;
  unrealized_pnl: string | null;
  win_rate: string | null;
  avg_trade_return: string | null;
  profit_factor: string | null;
  cagr: string | null;
  roi_30d: string | null;
  roi_90d: string | null;
  monthly_consistency: string | null;
  weekly_consistency: string | null;
  max_drawdown: string | null;
  losing_streak_max: number | null;
  volatility: string | null;
  concentration: string | null;
  sharpe: string | null;
  sortino: string | null;
  avg_hold_hours: string | null;
  trade_frequency: string | null;
  avg_position_size: string | null;
  max_concurrent_positions: number | null;
  category_mix: Record<string, number> | null;
  early_entry_score: string | null;
  exit_timing_score: string | null;
  avg_prob_improvement: string | null;
  sample_size: number | null;
}

export interface WalletSummary {
  address: string;
  first_seen: string | null;
  last_active: string | null;
  discovery_source: string | null;
  discovered_at: string | null;
  total_volume: string | null;
  total_trades: number | null;
  is_tracked: boolean;
  copy_enabled: boolean;
  disabled_reason: string | null;
}

export interface Wallet extends WalletSummary {
  metrics: WalletMetrics | null;
}

export interface WalletListResponse {
  items: Wallet[];
  total: number;
  limit: number;
  offset: number;
}

export interface WalletTrade {
  id: number;
  market_id: string | null;
  token_id: string | null;
  side: "BUY" | "SELL";
  price: string;
  size: string;
  usdc_amount: string | null;
  timestamp: string;
  tx_hash: string;
}

export interface WalletPosition {
  id: number;
  market_id: string | null;
  token_id: string | null;
  net_size: string;
  avg_entry_price: string | null;
  realized_pnl: string | null;
  unrealized_pnl: string | null;
  is_open: boolean;
  opened_at: string | null;
  closed_at: string | null;
}

export interface WalletDetail extends Wallet {
  open_positions: WalletPosition[];
  recent_trades: WalletTrade[];
}

export type WalletSortKey =
  | "total_volume"
  | "total_trades"
  | "last_active"
  | "discovered_at"
  | "roi_lifetime"
  | "realized_pnl"
  | "win_rate"
  | "sharpe";

export interface WalletListParams {
  tracked?: boolean;
  source?: string;
  search?: string;
  sort?: WalletSortKey;
  order?: "asc" | "desc";
  limit?: number;
  offset?: number;
}

export function getWallets(params: WalletListParams = {}): Promise<WalletListResponse> {
  const query = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value !== undefined && value !== "") query.set(key, String(value));
  }
  const qs = query.toString();
  return apiFetch<WalletListResponse>(`/wallets${qs ? `?${qs}` : ""}`);
}

export function getWallet(address: string): Promise<WalletDetail> {
  return apiFetch<WalletDetail>(`/wallets/${address}`);
}

export function refreshWalletMetrics(address: string): Promise<Wallet> {
  return apiFetch<Wallet>(`/wallets/${address}/refresh-metrics`, { method: "POST" });
}

export function backfillWallet(address: string): Promise<{ task_id: string }> {
  return apiFetch<{ task_id: string }>(`/wallets/${address}/backfill`, { method: "POST" });
}

export interface DiscoveryStatus {
  total_wallets: number;
  tracked_wallets: number;
  last_discovered_at: string | null;
  by_source: { source: string | null; count: number }[];
}

export function getDiscoveryStatus(): Promise<DiscoveryStatus> {
  return apiFetch<DiscoveryStatus>("/discovery");
}

export function runDiscovery(): Promise<{ task_id: string }> {
  return apiFetch<{ task_id: string }>("/discovery/run", { method: "POST" });
}

export type RankingCategory =
  | "overall"
  | "politics"
  | "crypto"
  | "sports"
  | "economics"
  | "hft"
  | "long_term"
  | "early_info"
  | "low_risk";

export interface ScoreExplanationInput {
  value: number;
  score: number;
}

export interface ScoreExplanationComponent {
  score: number;
  weight: number;
  contribution: number;
  neutral_prior?: boolean;
  inputs: Record<string, ScoreExplanationInput>;
}

export interface ScoreExplanation {
  cohort_size: number;
  components: Record<string, ScoreExplanationComponent>;
}

export interface WalletScoreDetail {
  computed_at: string;
  composite: string;
  historical_component: string | null;
  recent_component: string | null;
  copy_quality_component: string | null;
  risk_component: string | null;
  explanation: ScoreExplanation | null;
}

export interface RankingEntry {
  rank: number;
  score: string;
  wallet: WalletSummary;
  score_detail: WalletScoreDetail | null;
}

export interface RankingsResponse {
  computed_at: string | null;
  category: RankingCategory;
  categories: { category: RankingCategory; count: number }[];
  items: RankingEntry[];
}

export function getRankings(
  category: RankingCategory = "overall",
  limit = 50,
): Promise<RankingsResponse> {
  return apiFetch<RankingsResponse>(`/rankings?category=${category}&limit=${limit}`);
}

export interface RebuildRankingsResponse {
  computed_at: string;
  wallets_scored: number;
  categories: Record<string, number>;
}

export function rebuildRankings(): Promise<RebuildRankingsResponse> {
  return apiFetch<RebuildRankingsResponse>("/rankings/rebuild", { method: "POST" });
}

export type BacktestStatus = "queued" | "running" | "completed" | "failed";
export type SizingMode = "fixed" | "percentage" | "proportional";

// Decimal fields are sent as strings (parsed exactly by the backend).
export interface BacktestConfig {
  wallets: string[];
  start?: string | null;
  end?: string | null;
  initial_capital?: string;
  sizing_mode?: SizingMode;
  sizing_value?: string;
  weights?: Record<string, string> | null;
  delay_seconds?: number;
  slippage_pct?: string;
  fee_pct?: string;
  max_open_positions?: number | null;
  max_position_size?: string | null;
}

export interface BacktestMetrics {
  initial_capital: number;
  final_equity: number;
  total_return: number;
  cagr: number | null;
  max_drawdown: number | null;
  sharpe: number | null;
  sortino: number | null;
  win_rate: number | null;
  profit_factor: number | null;
  turnover: number;
  fees_paid: number;
  trades_executed: number;
  trades_skipped: number;
  error?: string;
}

export interface EquityPoint {
  t: string;
  equity: number;
}

export interface BacktestTradeLogEntry {
  t: string;
  wallet: string;
  token: string;
  market_id: string | null;
  side: "BUY" | "SELL";
  source_price: number;
  source_size: number;
  status: "filled" | "skipped";
  reason: string | null;
  exec_price: number | null;
  size: number | null;
  notional: number | null;
  fee: number | null;
  realized_pnl: number | null;
  cash: number;
  equity: number;
}

export interface Backtest {
  id: number;
  status: BacktestStatus;
  params: BacktestConfig & Record<string, unknown>;
  started_at: string | null;
  finished_at: string | null;
  metrics: BacktestMetrics | null;
}

export interface BacktestDetail extends Backtest {
  equity_curve: EquityPoint[] | null;
  trade_log: BacktestTradeLogEntry[] | null;
}

export interface BacktestListResponse {
  items: Backtest[];
  total: number;
  limit: number;
  offset: number;
}

export function createBacktest(config: BacktestConfig): Promise<Backtest> {
  return apiFetch<Backtest>("/backtests", { method: "POST", body: JSON.stringify(config) });
}

export function getBacktests(limit = 20, offset = 0): Promise<BacktestListResponse> {
  return apiFetch<BacktestListResponse>(`/backtests?limit=${limit}&offset=${offset}`);
}

export function getBacktest(id: number): Promise<BacktestDetail> {
  return apiFetch<BacktestDetail>(`/backtests/${id}`);
}

// --- Phase 5: risk, copy, portfolio (Decimals arrive as strings) --------------------------

export interface RiskLimits {
  max_total_exposure_pct: string;
  max_daily_loss: string;
  max_weekly_loss: string;
  max_monthly_loss: string;
  max_open_positions: number;
  max_position_size_pct: string;
  max_per_market_exposure_pct: string;
  max_per_wallet_exposure_pct: string;
  min_market_liquidity: string;
  max_slippage_pct: string;
  max_copy_delay_seconds: number;
}

export interface KillSwitch {
  engaged: boolean;
  reason: string | null;
  updated_at: string | null;
}

export interface RiskEvent {
  id: number;
  timestamp: string;
  type: string;
  severity: string;
  context: Record<string, unknown> | null;
  action_taken: string | null;
}

export interface RiskStatus {
  kill_switch: KillSwitch;
  limits: RiskLimits;
  execution_mode: string;
  recent_events: RiskEvent[];
}

export interface RiskEventList {
  items: RiskEvent[];
  total: number;
  limit: number;
  offset: number;
}

export function getRiskStatus(): Promise<RiskStatus> {
  return apiFetch<RiskStatus>("/risk");
}

export function getRiskEvents(limit = 50, offset = 0): Promise<RiskEventList> {
  return apiFetch<RiskEventList>(`/risk/events?limit=${limit}&offset=${offset}`);
}

export function setKillSwitch(enabled: boolean, reason?: string): Promise<KillSwitch> {
  return apiFetch<KillSwitch>("/risk/kill-switch", {
    method: "POST",
    body: JSON.stringify({ enabled, reason }),
  });
}

export type CopyOrderStatus = "pending" | "filled" | "partial" | "rejected" | "skipped";

export interface CopyOrder {
  id: number;
  created_at: string;
  mode: string;
  source_wallet: string | null;
  source_trade_id: number | null;
  market_id: string | null;
  token_id: string | null;
  side: "BUY" | "SELL";
  requested_size: string;
  filled_size: string | null;
  requested_price: string | null;
  fill_price: string | null;
  delay_ms: number | null;
  slippage: string | null;
  status: CopyOrderStatus;
  error: string | null;
}

export interface CopyOrderList {
  items: CopyOrder[];
  total: number;
  limit: number;
  offset: number;
}

export interface CopyStatus {
  mode: string;
  total_orders: number;
  by_status: Record<string, number>;
  last_order_at: string | null;
}

export interface SimulateTradeBody {
  source_wallet: string;
  token_id: string;
  side: "BUY" | "SELL";
  size: string;
  price: string;
  market_id?: string | null;
  usdc_amount?: string | null;
}

export interface SimulateResult {
  order: CopyOrder;
  detail: string;
}

export function getCopyStatus(mode = "paper"): Promise<CopyStatus> {
  return apiFetch<CopyStatus>(`/copy?mode=${mode}`);
}

export function getCopyOrders(mode = "paper", limit = 50): Promise<CopyOrderList> {
  return apiFetch<CopyOrderList>(`/copy/orders?mode=${mode}&limit=${limit}`);
}

export function simulateTrade(body: SimulateTradeBody): Promise<SimulateResult> {
  return apiFetch<SimulateResult>("/copy/simulate", {
    method: "POST",
    body: JSON.stringify(body),
  });
}

export function runDetector(): Promise<{ task_id: string }> {
  return apiFetch<{ task_id: string }>("/copy/run", { method: "POST" });
}

export interface PortfolioPosition {
  id: number;
  mode: string;
  market_id: string | null;
  token_id: string | null;
  source_wallet: string | null;
  size: string;
  avg_entry_price: string | null;
  realized_pnl: string | null;
  unrealized_pnl: string | null;
  is_open: boolean;
  opened_at: string | null;
  closed_at: string | null;
}

export interface PortfolioSnapshot {
  id: number;
  timestamp: string;
  mode: string;
  cash: string;
  total_value: string;
  realized_pnl: string | null;
  unrealized_pnl: string | null;
  open_positions: number | null;
}

export interface PortfolioSummary {
  mode: string;
  cash: string;
  total_value: string;
  realized_pnl: string;
  unrealized_pnl: string;
  open_positions: number;
  last_snapshot_at: string | null;
  positions: PortfolioPosition[];
}

export interface PortfolioHistory {
  mode: string;
  items: PortfolioSnapshot[];
}

export function getPortfolio(mode = "paper"): Promise<PortfolioSummary> {
  return apiFetch<PortfolioSummary>(`/portfolio?mode=${mode}`);
}

export function getPortfolioHistory(mode = "paper"): Promise<PortfolioHistory> {
  return apiFetch<PortfolioHistory>(`/portfolio/history?mode=${mode}`);
}

export function revaluePortfolio(mode = "paper"): Promise<PortfolioSnapshot> {
  return apiFetch<PortfolioSnapshot>(`/portfolio/revalue?mode=${mode}`, { method: "POST" });
}

export interface ClosePositionResult {
  status: string;
  filled_size: string;
  fill_price: string | null;
  error: string | null;
  position: PortfolioPosition;
}

export function closePosition(positionId: number): Promise<ClosePositionResult> {
  return apiFetch<ClosePositionResult>(`/portfolio/positions/${positionId}/close`, {
    method: "POST",
  });
}

// Shape pushed by the /ws/live socket every interval.
export interface LiveFrame {
  type: "state";
  portfolio: PortfolioSummary;
  kill_switch: KillSwitch;
  copy_orders: CopyOrder[];
  risk_events: RiskEvent[];
}

// --- Phase 6: strategy optimizer ----------------------------------------------------------
export type Weighting = "equal" | "score" | "volatility";

// The re-runnable strategy spec stored in `strategies.params`.
export interface StrategySpec {
  category: string;
  top_n: number;
  weighting: Weighting;
  sizing_mode: SizingMode;
  sizing_value: string;
  delay_seconds: number;
  slippage_pct: string;
  fee_pct: string;
  max_open_positions: number | null;
  max_position_size: string | null;
}

// One candidate's compact grade inside a run's rationale.
export interface CandidateSummary {
  name: string;
  wallets: number;
  objective_value: number | null;
  passes_risk: boolean;
  total_return: number | null;
  sharpe: number | null;
  max_drawdown: number | null;
  trades_executed: number | null;
}

export type OptimizerDecision = "adopted" | "kept_active" | "no_candidate";

export interface StrategyRationale {
  decision: OptimizerDecision;
  adopted: boolean;
  objective: string;
  min_improvement: number;
  max_drawdown_cap: number;
  lookback_days: number;
  summary: string;
  chosen: CandidateSummary | null;
  active: CandidateSummary | null;
  candidates: CandidateSummary[];
}

export interface Strategy {
  id: number;
  name: string;
  is_active: boolean;
  params: StrategySpec & Record<string, unknown>;
  created_at: string;
  adopted_at: string | null;
}

export interface StrategyDetail extends Strategy {
  rationale: StrategyRationale | null;
}

export interface StrategyListResponse {
  items: Strategy[];
  total: number;
  limit: number;
  offset: number;
}

export function getStrategies(limit = 20, offset = 0): Promise<StrategyListResponse> {
  return apiFetch<StrategyListResponse>(`/strategies?limit=${limit}&offset=${offset}`);
}

export function getStrategy(id: number): Promise<StrategyDetail> {
  return apiFetch<StrategyDetail>(`/strategies/${id}`);
}

// The active strategy 404s until the first adoption; callers treat that as "none yet".
export function getActiveStrategy(): Promise<StrategyDetail> {
  return apiFetch<StrategyDetail>("/strategies/active");
}

export function runOptimizer(): Promise<{ task_id: string; message: string }> {
  return apiFetch<{ task_id: string; message: string }>("/strategies/optimize", { method: "POST" });
}
