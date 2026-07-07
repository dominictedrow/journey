"use client";

import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { EquityChart, MAX_COMPARE, SERIES_COLORS } from "@/components/charts/EquityChart";
import {
  createBacktest,
  getBacktest,
  getBacktests,
  getWallets,
  type Backtest,
  type BacktestConfig,
  type BacktestDetail,
  type BacktestMetrics,
  type SizingMode,
} from "@/lib/api";
import { fmtDate, fmtNum, fmtPct, fmtUsd, pnlClass, shortAddress } from "@/lib/format";

const ACTIVE_STATUSES = new Set(["queued", "running"]);

const SIZING_HELP: Record<SizingMode, string> = {
  fixed: "USDC spent per copied buy",
  percentage: "fraction of equity per copied buy (0.1 = 10%)",
  proportional: "multiple of the source trade's notional",
};

function statusClass(status: string): string {
  if (status === "completed") return "bg-emerald-900 text-emerald-300";
  if (status === "failed") return "bg-red-900 text-red-300";
  return "bg-amber-900 text-amber-300";
}

function MetricTile({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="rounded border border-slate-800 bg-slate-950 px-3 py-2">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`mt-0.5 font-mono text-sm ${tone ?? "text-slate-200"}`}>{value}</p>
    </div>
  );
}

function MetricsGrid({ metrics }: { metrics: BacktestMetrics }) {
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-5">
      <MetricTile
        label="Final equity"
        value={fmtUsd(metrics.final_equity)}
        tone={pnlClass(metrics.total_return)}
      />
      <MetricTile
        label="Total return"
        value={fmtPct(metrics.total_return)}
        tone={pnlClass(metrics.total_return)}
      />
      <MetricTile label="CAGR" value={fmtPct(metrics.cagr)} />
      <MetricTile label="Max drawdown" value={fmtPct(metrics.max_drawdown)} />
      <MetricTile label="Sharpe" value={fmtNum(metrics.sharpe)} />
      <MetricTile label="Sortino" value={fmtNum(metrics.sortino)} />
      <MetricTile label="Win rate" value={fmtPct(metrics.win_rate)} />
      <MetricTile label="Profit factor" value={fmtNum(metrics.profit_factor)} />
      <MetricTile label="Turnover" value={fmtNum(metrics.turnover)} />
      <MetricTile
        label="Fills / skips"
        value={`${metrics.trades_executed} / ${metrics.trades_skipped}`}
      />
    </div>
  );
}

function exportBacktest(detail: BacktestDetail) {
  const blob = new Blob([JSON.stringify(detail, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `backtest-${detail.id}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

export default function BacktestingPage() {
  const queryClient = useQueryClient();
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [compare, setCompare] = useState<{ id: number; color: string }[]>([]);

  const [walletSel, setWalletSel] = useState<string[]>([]);
  const [capital, setCapital] = useState("1000");
  const [sizingMode, setSizingMode] = useState<SizingMode>("fixed");
  const [sizingValue, setSizingValue] = useState("50");
  const [delaySeconds, setDelaySeconds] = useState("5");
  const [slippagePct, setSlippagePct] = useState("0.5");
  const [feePct, setFeePct] = useState("0");

  const wallets = useQuery({
    queryKey: ["wallets", { tracked: true, forBacktest: true }],
    queryFn: () => getWallets({ tracked: true, limit: 200 }),
  });

  const list = useQuery({
    queryKey: ["backtests"],
    queryFn: () => getBacktests(50),
    refetchInterval: (query) =>
      query.state.data?.items.some((b) => ACTIVE_STATUSES.has(b.status)) ? 2000 : false,
  });

  const shownRuns: { id: number; color: string }[] =
    compare.length > 0
      ? compare
      : selectedId !== null
        ? [{ id: selectedId, color: SERIES_COLORS[0] }]
        : [];

  const details = useQueries({
    queries: shownRuns.map(({ id }) => ({
      queryKey: ["backtest", id],
      queryFn: () => getBacktest(id),
      refetchInterval: (query: { state: { data?: BacktestDetail } }) =>
        query.state.data && ACTIVE_STATUSES.has(query.state.data.status) ? 2000 : false,
    })),
  });

  const run = useMutation({
    mutationFn: (config: BacktestConfig) => createBacktest(config),
    onSuccess: (created) => {
      queryClient.invalidateQueries({ queryKey: ["backtests"] });
      setCompare([]);
      setSelectedId(created.id);
    },
  });

  const submit = () => {
    run.mutate({
      wallets: walletSel,
      initial_capital: capital,
      sizing_mode: sizingMode,
      sizing_value: sizingValue,
      delay_seconds: Number(delaySeconds) || 0,
      slippage_pct: String((Number(slippagePct) || 0) / 100),
      fee_pct: String((Number(feePct) || 0) / 100),
    });
  };

  const toggleCompare = (backtest: Backtest) => {
    setCompare((current) => {
      if (current.some((c) => c.id === backtest.id)) {
        return current.filter((c) => c.id !== backtest.id);
      }
      if (current.length >= MAX_COMPARE) return current;
      // Color follows the run while selected — freed slots are reused, survivors keep theirs.
      const used = new Set(current.map((c) => c.color));
      const color = SERIES_COLORS.find((c) => !used.has(c)) ?? SERIES_COLORS[0];
      return [...current, { id: backtest.id, color }];
    });
  };

  const series = shownRuns.flatMap(({ id, color }) => {
    const detail = details.find((d) => d.data?.id === id)?.data;
    if (!detail?.equity_curve) return [];
    return [
      {
        key: String(id),
        name: `Run #${id}`,
        color,
        points: detail.equity_curve.map((p) => ({ t: Date.parse(p.t), equity: p.equity })),
      },
    ];
  });

  const primary = compare.length === 0 && selectedId !== null ? details[0]?.data : undefined;

  return (
    <div>
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Backtesting</h1>
        {list.data && <p className="text-sm text-slate-500">{list.data.total} runs</p>}
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[20rem,1fr]">
        {/* Config form */}
        <div className="rounded border border-slate-800 bg-slate-950 p-4">
          <h2 className="text-sm font-medium text-slate-200">New backtest</h2>
          <p className="mt-2 text-xs text-slate-500">Source wallets (tracked)</p>
          <div className="mt-1 max-h-40 overflow-y-auto rounded border border-slate-800">
            {wallets.data?.items.map((w) => (
              <label
                key={w.address}
                className="flex items-center gap-2 px-2 py-1 text-xs text-slate-300 hover:bg-slate-900"
              >
                <input
                  type="checkbox"
                  checked={walletSel.includes(w.address)}
                  onChange={(e) =>
                    setWalletSel((s) =>
                      e.target.checked ? [...s, w.address] : s.filter((a) => a !== w.address),
                    )
                  }
                />
                <span className="font-mono">{shortAddress(w.address)}</span>
                <span className="ml-auto text-slate-500">{w.total_trades ?? 0} trades</span>
              </label>
            ))}
            {wallets.data?.items.length === 0 && (
              <p className="px-2 py-3 text-xs text-slate-500">
                No tracked wallets — track some in the Wallets tab first.
              </p>
            )}
          </div>

          <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
            <label className="text-slate-400">
              Capital (USDC)
              <input
                value={capital}
                onChange={(e) => setCapital(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100"
              />
            </label>
            <label className="text-slate-400">
              Sizing mode
              <select
                value={sizingMode}
                onChange={(e) => setSizingMode(e.target.value as SizingMode)}
                className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100"
              >
                <option value="fixed">Fixed</option>
                <option value="percentage">Percentage</option>
                <option value="proportional">Proportional</option>
              </select>
            </label>
            <label className="col-span-2 text-slate-400">
              Sizing value <span className="text-slate-600">({SIZING_HELP[sizingMode]})</span>
              <input
                value={sizingValue}
                onChange={(e) => setSizingValue(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100"
              />
            </label>
            <label className="text-slate-400">
              Delay (s)
              <input
                value={delaySeconds}
                onChange={(e) => setDelaySeconds(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100"
              />
            </label>
            <label className="text-slate-400">
              Slippage (%)
              <input
                value={slippagePct}
                onChange={(e) => setSlippagePct(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100"
              />
            </label>
            <label className="text-slate-400">
              Fee (%)
              <input
                value={feePct}
                onChange={(e) => setFeePct(e.target.value)}
                className="mt-1 w-full rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100"
              />
            </label>
          </div>

          <button
            onClick={submit}
            disabled={walletSel.length === 0 || run.isPending}
            className="mt-4 w-full rounded bg-sky-700 px-3 py-2 text-sm font-medium text-white hover:bg-sky-600 disabled:opacity-40"
          >
            {run.isPending ? "Queuing…" : "Run backtest"}
          </button>
          {run.error && <p className="mt-2 text-xs text-red-400">{run.error.message}</p>}
        </div>

        {/* Runs list */}
        <div className="overflow-x-auto rounded border border-slate-800">
          <table className="w-full text-sm">
            <thead className="bg-slate-900 text-left text-slate-400">
              <tr>
                <th className="px-3 py-2 font-medium">Run</th>
                <th className="px-3 py-2 font-medium">Status</th>
                <th className="px-3 py-2 font-medium">Wallets</th>
                <th className="px-3 py-2 font-medium">Final equity</th>
                <th className="px-3 py-2 font-medium">Return</th>
                <th className="px-3 py-2 font-medium">Max DD</th>
                <th className="px-3 py-2 font-medium">Finished</th>
                <th className="px-3 py-2 font-medium">Compare</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {list.data?.items.map((b) => (
                <tr
                  key={b.id}
                  onClick={() => {
                    setSelectedId(b.id);
                    setCompare([]);
                  }}
                  className={`cursor-pointer hover:bg-slate-900/50 ${
                    selectedId === b.id && compare.length === 0 ? "bg-slate-900/70" : ""
                  }`}
                >
                  <td className="px-3 py-2 font-mono text-sky-400">#{b.id}</td>
                  <td className="px-3 py-2">
                    <span
                      className={`rounded px-2 py-0.5 text-xs font-medium ${statusClass(b.status)}`}
                    >
                      {b.status}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-slate-400">{b.params.wallets?.length ?? 0}</td>
                  <td className="px-3 py-2">{fmtUsd(b.metrics?.final_equity)}</td>
                  <td className={`px-3 py-2 ${pnlClass(b.metrics?.total_return)}`}>
                    {fmtPct(b.metrics?.total_return)}
                  </td>
                  <td className="px-3 py-2">{fmtPct(b.metrics?.max_drawdown)}</td>
                  <td className="px-3 py-2 text-slate-400">{fmtDate(b.finished_at)}</td>
                  <td className="px-3 py-2" onClick={(e) => e.stopPropagation()}>
                    {b.status === "completed" && (
                      <input
                        type="checkbox"
                        checked={compare.some((c) => c.id === b.id)}
                        disabled={
                          !compare.some((c) => c.id === b.id) && compare.length >= MAX_COMPARE
                        }
                        onChange={() => toggleCompare(b)}
                      />
                    )}
                  </td>
                </tr>
              ))}
              {!list.isPending && list.data?.items.length === 0 && (
                <tr>
                  <td colSpan={8} className="px-3 py-8 text-center text-slate-500">
                    No backtests yet — configure one on the left.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Result */}
      {series.length > 0 && (
        <div className="mt-6 rounded border border-slate-800 bg-slate-950 p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-slate-200">
              {compare.length > 0
                ? `Equity comparison (${series.length} runs)`
                : `Run #${selectedId} equity`}
            </h2>
            {primary && primary.status === "completed" && (
              <button
                onClick={() => exportBacktest(primary)}
                className="rounded bg-slate-800 px-3 py-1 text-xs text-slate-300 hover:bg-slate-700"
              >
                Export JSON
              </button>
            )}
          </div>
          <div className="mt-3">
            <EquityChart series={series} />
          </div>

          {primary?.metrics && !primary.metrics.error && (
            <div className="mt-4">
              <MetricsGrid metrics={primary.metrics} />
            </div>
          )}
          {primary?.metrics?.error && (
            <p className="mt-4 rounded border border-red-900 bg-red-950 p-3 text-sm text-red-300">
              Backtest failed: {primary.metrics.error}
            </p>
          )}

          {primary?.trade_log && primary.trade_log.length > 0 && (
            <div className="mt-4">
              <h3 className="text-xs font-medium text-slate-400">
                Trade log ({primary.trade_log.length} events, first 50 shown)
              </h3>
              <div className="mt-2 max-h-72 overflow-auto rounded border border-slate-800">
                <table className="w-full text-xs">
                  <thead className="sticky top-0 bg-slate-900 text-left text-slate-400">
                    <tr>
                      <th className="px-2 py-1.5 font-medium">Time</th>
                      <th className="px-2 py-1.5 font-medium">Wallet</th>
                      <th className="px-2 py-1.5 font-medium">Side</th>
                      <th className="px-2 py-1.5 font-medium">Src price</th>
                      <th className="px-2 py-1.5 font-medium">Exec price</th>
                      <th className="px-2 py-1.5 font-medium">Notional</th>
                      <th className="px-2 py-1.5 font-medium">PnL</th>
                      <th className="px-2 py-1.5 font-medium">Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-800">
                    {primary.trade_log.slice(0, 50).map((e, i) => (
                      <tr key={i} className="text-slate-300">
                        <td className="px-2 py-1 text-slate-400">{fmtDate(e.t)}</td>
                        <td className="px-2 py-1 font-mono">{shortAddress(e.wallet)}</td>
                        <td
                          className={
                            e.side === "BUY" ? "px-2 py-1 text-sky-400" : "px-2 py-1 text-amber-400"
                          }
                        >
                          {e.side}
                        </td>
                        <td className="px-2 py-1">{fmtNum(e.source_price, 4)}</td>
                        <td className="px-2 py-1">{fmtNum(e.exec_price, 4)}</td>
                        <td className="px-2 py-1">{fmtUsd(e.notional)}</td>
                        <td className={`px-2 py-1 ${pnlClass(e.realized_pnl)}`}>
                          {e.realized_pnl === null ? "—" : fmtUsd(e.realized_pnl)}
                        </td>
                        <td className="px-2 py-1 text-slate-400">
                          {e.status === "filled" ? "filled" : `skipped: ${e.reason}`}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
