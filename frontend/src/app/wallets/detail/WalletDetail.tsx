"use client";

import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useSearchParams } from "next/navigation";

import { getWallet, type WalletMetrics } from "@/lib/api";
import { fmtDate, fmtNum, fmtPct, fmtUsd, pnlClass } from "@/lib/format";

function Metric({ label, value, className }: { label: string; value: string; className?: string }) {
  return (
    <div className="rounded border border-slate-800 bg-slate-900/50 p-3">
      <div className="text-xs text-slate-500">{label}</div>
      <div className={`mt-1 text-lg font-semibold ${className ?? "text-slate-100"}`}>{value}</div>
    </div>
  );
}

function MetricsGrid({ m }: { m: WalletMetrics }) {
  return (
    <div className="space-y-6">
      <section>
        <h2 className="text-sm font-medium uppercase tracking-wide text-slate-500">Performance</h2>
        <div className="mt-2 grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
          <Metric label="Lifetime ROI" value={fmtPct(m.roi_lifetime)} className={pnlClass(m.roi_lifetime)} />
          <Metric label="Realized PnL" value={fmtUsd(m.realized_pnl)} className={pnlClass(m.realized_pnl)} />
          <Metric label="Unrealized PnL" value={fmtUsd(m.unrealized_pnl)} className={pnlClass(m.unrealized_pnl)} />
          <Metric label="Win rate" value={fmtPct(m.win_rate)} />
          <Metric label="Profit factor" value={fmtNum(m.profit_factor)} />
          <Metric label="CAGR" value={fmtPct(m.cagr)} className={pnlClass(m.cagr)} />
        </div>
      </section>
      <section>
        <h2 className="text-sm font-medium uppercase tracking-wide text-slate-500">Recent form</h2>
        <div className="mt-2 grid grid-cols-2 gap-3 md:grid-cols-4">
          <Metric label="ROI 30d" value={fmtPct(m.roi_30d)} className={pnlClass(m.roi_30d)} />
          <Metric label="ROI 90d" value={fmtPct(m.roi_90d)} className={pnlClass(m.roi_90d)} />
          <Metric label="Positive months" value={fmtPct(m.monthly_consistency)} />
          <Metric label="Positive weeks" value={fmtPct(m.weekly_consistency)} />
        </div>
      </section>
      <section>
        <h2 className="text-sm font-medium uppercase tracking-wide text-slate-500">Risk</h2>
        <div className="mt-2 grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
          <Metric label="Max drawdown" value={fmtPct(m.max_drawdown)} />
          <Metric label="Worst losing streak" value={fmtNum(m.losing_streak_max, 0)} />
          <Metric label="Volatility" value={fmtNum(m.volatility)} />
          <Metric label="Sharpe" value={fmtNum(m.sharpe)} />
          <Metric label="Sortino" value={fmtNum(m.sortino)} />
          <Metric label="Concentration (HHI)" value={fmtNum(m.concentration)} />
        </div>
      </section>
      <section>
        <h2 className="text-sm font-medium uppercase tracking-wide text-slate-500">Behavior</h2>
        <div className="mt-2 grid grid-cols-2 gap-3 md:grid-cols-4 xl:grid-cols-6">
          <Metric label="Avg hold (hours)" value={fmtNum(m.avg_hold_hours, 1)} />
          <Metric label="Trades / day" value={fmtNum(m.trade_frequency)} />
          <Metric label="Avg position size" value={fmtUsd(m.avg_position_size)} />
          <Metric label="Max concurrent" value={fmtNum(m.max_concurrent_positions, 0)} />
          <Metric label="Early entry score" value={fmtPct(m.early_entry_score)} />
          <Metric label="Exit timing score" value={fmtPct(m.exit_timing_score)} />
        </div>
      </section>
      {m.category_mix && Object.keys(m.category_mix).length > 0 && (
        <section>
          <h2 className="text-sm font-medium uppercase tracking-wide text-slate-500">
            Category mix
          </h2>
          <div className="mt-2 space-y-2">
            {Object.entries(m.category_mix)
              .sort(([, a], [, b]) => b - a)
              .map(([category, share]) => (
                <div key={category} className="flex items-center gap-3 text-sm">
                  <span className="w-32 shrink-0 text-slate-400">{category}</span>
                  <div className="h-2 flex-1 rounded bg-slate-800">
                    <div
                      className="h-2 rounded bg-sky-600"
                      style={{ width: `${Math.min(share * 100, 100)}%` }}
                    />
                  </div>
                  <span className="w-14 text-right text-slate-300">{fmtPct(share)}</span>
                </div>
              ))}
          </div>
        </section>
      )}
      <p className="text-xs text-slate-600">
        Computed {fmtDate(m.computed_at)} from {m.sample_size ?? 0} indexed fills. Avg probability
        edge per round-trip: {fmtNum(m.avg_prob_improvement, 3)}.
      </p>
    </div>
  );
}

export function WalletDetail() {
  const address = useSearchParams().get("address") ?? "";

  const { data: wallet, isPending, error } = useQuery({
    queryKey: ["wallet", address],
    queryFn: () => getWallet(address),
    enabled: address !== "",
  });

  if (address === "")
    return (
      <p className="rounded border border-slate-800 bg-slate-900/50 p-3 text-sm text-slate-400">
        No wallet selected. <Link href="/wallets" className="text-sky-400 hover:underline">Browse wallets</Link>.
      </p>
    );
  if (isPending) return <p className="text-slate-500">Loading wallet…</p>;
  if (error || !wallet)
    return (
      <p className="rounded border border-red-900 bg-red-950 p-3 text-sm text-red-300">
        Failed to load wallet: {error?.message ?? "not found"}
      </p>
    );

  return (
    <div>
      <Link href="/wallets" className="text-sm text-slate-500 hover:text-slate-300">
        ← Wallets
      </Link>
      <div className="mt-2 flex flex-wrap items-center gap-3">
        <h1 className="font-mono text-xl font-semibold">{wallet.address}</h1>
        <span
          className={`rounded px-3 py-1 text-sm font-medium ${
            wallet.is_tracked
              ? "bg-emerald-900 text-emerald-300"
              : "bg-slate-800 text-slate-400"
          }`}
        >
          {wallet.is_tracked ? "Tracked" : "Not tracked"}
        </span>
        <span
          className={`rounded px-3 py-1 text-sm font-medium ${
            wallet.copy_enabled
              ? "bg-sky-900 text-sky-300"
              : "bg-slate-800 text-slate-400"
          }`}
        >
          {wallet.copy_enabled ? "Copy enabled" : "Copy disabled"}
        </span>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        {wallet.discovery_source ? `Discovered via ${wallet.discovery_source}` : "Seen on-chain"}
        {wallet.discovered_at ? ` on ${fmtDate(wallet.discovered_at)}` : ""} · Active{" "}
        {fmtDate(wallet.first_seen)} → {fmtDate(wallet.last_active)}
      </p>

      <div className="mt-8">
        {wallet.metrics ? (
          <MetricsGrid m={wallet.metrics} />
        ) : (
          <p className="rounded border border-slate-800 bg-slate-900/50 p-4 text-sm text-slate-400">
            No metrics yet — on-chain history and metrics compute automatically once this wallet
            is tracked.
          </p>
        )}
      </div>

      <section className="mt-10">
        <h2 className="text-lg font-semibold">Open positions ({wallet.open_positions.length})</h2>
        <div className="mt-3 overflow-x-auto rounded border border-slate-800">
          <table className="w-full text-sm">
            <thead className="bg-slate-900 text-left text-slate-400">
              <tr>
                <th className="px-3 py-2 font-medium">Market</th>
                <th className="px-3 py-2 font-medium">Size</th>
                <th className="px-3 py-2 font-medium">Avg entry</th>
                <th className="px-3 py-2 font-medium">Realized PnL</th>
                <th className="px-3 py-2 font-medium">Opened</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {wallet.open_positions.map((p) => (
                <tr key={p.id}>
                  <td className="max-w-xs truncate px-3 py-2 font-mono text-xs text-slate-400">
                    {p.market_id ?? p.token_id ?? "—"}
                  </td>
                  <td className="px-3 py-2">{fmtNum(p.net_size, 2)}</td>
                  <td className="px-3 py-2">{fmtNum(p.avg_entry_price, 3)}</td>
                  <td className={`px-3 py-2 ${pnlClass(p.realized_pnl)}`}>
                    {fmtUsd(p.realized_pnl)}
                  </td>
                  <td className="px-3 py-2 text-slate-400">{fmtDate(p.opened_at)}</td>
                </tr>
              ))}
              {wallet.open_positions.length === 0 && (
                <tr>
                  <td colSpan={5} className="px-3 py-6 text-center text-slate-500">
                    No open positions.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      <section className="mt-10">
        <h2 className="text-lg font-semibold">Recent trades</h2>
        <div className="mt-3 overflow-x-auto rounded border border-slate-800">
          <table className="w-full text-sm">
            <thead className="bg-slate-900 text-left text-slate-400">
              <tr>
                <th className="px-3 py-2 font-medium">Time</th>
                <th className="px-3 py-2 font-medium">Side</th>
                <th className="px-3 py-2 font-medium">Price</th>
                <th className="px-3 py-2 font-medium">Size</th>
                <th className="px-3 py-2 font-medium">Notional</th>
                <th className="px-3 py-2 font-medium">Market</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800">
              {wallet.recent_trades.map((t) => (
                <tr key={t.id}>
                  <td className="px-3 py-2 text-slate-400">{fmtDate(t.timestamp)}</td>
                  <td
                    className={`px-3 py-2 font-medium ${
                      t.side === "BUY" ? "text-emerald-400" : "text-red-400"
                    }`}
                  >
                    {t.side}
                  </td>
                  <td className="px-3 py-2">{fmtNum(t.price, 3)}</td>
                  <td className="px-3 py-2">{fmtNum(t.size, 2)}</td>
                  <td className="px-3 py-2">{fmtUsd(t.usdc_amount)}</td>
                  <td className="max-w-xs truncate px-3 py-2 font-mono text-xs text-slate-400">
                    {t.market_id ?? t.token_id ?? "—"}
                  </td>
                </tr>
              ))}
              {wallet.recent_trades.length === 0 && (
                <tr>
                  <td colSpan={6} className="px-3 py-6 text-center text-slate-500">
                    No indexed trades yet.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
