"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { EquityChart, SERIES_COLORS } from "@/components/charts/EquityChart";
import {
  closePosition,
  getCopyOrders,
  getPortfolio,
  getPortfolioHistory,
  revaluePortfolio,
  type CopyOrder,
  type PortfolioPosition,
  type PortfolioSummary,
} from "@/lib/api";
import { fmtDate, fmtNum, fmtUsd, pnlClass, shortAddress } from "@/lib/format";
import { useLiveState } from "@/lib/ws";

function Tile({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="rounded border border-slate-800 bg-slate-950 px-3 py-2">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`mt-0.5 font-mono text-lg ${tone ?? "text-slate-100"}`}>{value}</p>
    </div>
  );
}

function statusClass(status: string): string {
  if (status === "filled") return "bg-emerald-900 text-emerald-300";
  if (status === "partial") return "bg-amber-900 text-amber-300";
  if (status === "rejected") return "bg-red-900 text-red-300";
  return "bg-slate-800 text-slate-400";
}

function SummaryTiles({ s }: { s: PortfolioSummary }) {
  const total = Number(s.total_value);
  return (
    <div className="grid grid-cols-2 gap-2 sm:grid-cols-3 xl:grid-cols-5">
      <Tile label="Total value" value={fmtUsd(total)} />
      <Tile label="Cash" value={fmtUsd(s.cash)} />
      <Tile label="Realized PnL" value={fmtUsd(s.realized_pnl)} tone={pnlClass(s.realized_pnl)} />
      <Tile
        label="Unrealized PnL"
        value={fmtUsd(s.unrealized_pnl)}
        tone={pnlClass(s.unrealized_pnl)}
      />
      <Tile label="Open positions" value={String(s.open_positions)} />
    </div>
  );
}

function PositionsTable({
  positions,
  onClose,
  closingId,
}: {
  positions: PortfolioPosition[];
  onClose: (id: number) => void;
  closingId: number | null;
}) {
  return (
    <div className="overflow-x-auto rounded border border-slate-800">
      <table className="w-full text-sm">
        <thead className="bg-slate-900 text-left text-slate-400">
          <tr>
            <th className="px-3 py-2 font-medium">Token</th>
            <th className="px-3 py-2 font-medium">Source</th>
            <th className="px-3 py-2 font-medium">Size</th>
            <th className="px-3 py-2 font-medium">Avg entry</th>
            <th className="px-3 py-2 font-medium">Unrealized</th>
            <th className="px-3 py-2 font-medium">Opened</th>
            <th className="px-3 py-2 font-medium" />
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {positions.map((p) => (
            <tr key={p.id} className="hover:bg-slate-900/50">
              <td className="px-3 py-2 font-mono text-slate-300">
                {p.token_id ? shortAddress(p.token_id) : "—"}
              </td>
              <td className="px-3 py-2 font-mono text-slate-400">
                {p.source_wallet ? shortAddress(p.source_wallet) : "—"}
              </td>
              <td className="px-3 py-2">{fmtNum(p.size)}</td>
              <td className="px-3 py-2">{fmtNum(p.avg_entry_price, 4)}</td>
              <td className={`px-3 py-2 ${pnlClass(p.unrealized_pnl)}`}>
                {fmtUsd(p.unrealized_pnl)}
              </td>
              <td className="px-3 py-2 text-slate-400">{fmtDate(p.opened_at)}</td>
              <td className="px-3 py-2 text-right">
                <button
                  onClick={() => onClose(p.id)}
                  disabled={closingId === p.id}
                  className="rounded bg-red-900/70 px-2 py-1 text-xs font-medium text-red-200 hover:bg-red-800 disabled:opacity-40"
                >
                  {closingId === p.id ? "Closing…" : "Close"}
                </button>
              </td>
            </tr>
          ))}
          {positions.length === 0 && (
            <tr>
              <td colSpan={7} className="px-3 py-8 text-center text-slate-500">
                No open positions — fire a copy from the Risk tab to populate the book.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

function OrdersTable({ orders }: { orders: CopyOrder[] }) {
  return (
    <div className="max-h-80 overflow-auto rounded border border-slate-800">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-slate-900 text-left text-slate-400">
          <tr>
            <th className="px-2 py-1.5 font-medium">Time</th>
            <th className="px-2 py-1.5 font-medium">Source</th>
            <th className="px-2 py-1.5 font-medium">Side</th>
            <th className="px-2 py-1.5 font-medium">Filled</th>
            <th className="px-2 py-1.5 font-medium">Price</th>
            <th className="px-2 py-1.5 font-medium">Slippage</th>
            <th className="px-2 py-1.5 font-medium">Status</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {orders.map((o) => (
            <tr key={o.id} className="text-slate-300">
              <td className="px-2 py-1 text-slate-400">{fmtDate(o.created_at)}</td>
              <td className="px-2 py-1 font-mono">
                {o.source_wallet ? shortAddress(o.source_wallet) : "—"}
              </td>
              <td
                className={o.side === "BUY" ? "px-2 py-1 text-sky-400" : "px-2 py-1 text-amber-400"}
              >
                {o.side}
              </td>
              <td className="px-2 py-1">{fmtNum(o.filled_size)}</td>
              <td className="px-2 py-1">{fmtNum(o.fill_price, 4)}</td>
              <td className="px-2 py-1">{o.slippage === null ? "—" : fmtNum(o.slippage, 4)}</td>
              <td className="px-2 py-1">
                <span className={`rounded px-2 py-0.5 ${statusClass(o.status)}`}>{o.status}</span>
                {o.error && <span className="ml-2 text-slate-500">{o.error}</span>}
              </td>
            </tr>
          ))}
          {orders.length === 0 && (
            <tr>
              <td colSpan={7} className="px-2 py-6 text-center text-slate-500">
                No copy orders yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export default function PortfolioPage() {
  const queryClient = useQueryClient();
  const { frame, connected } = useLiveState("paper");

  // The live socket carries summary + orders; queries are the fallback and the equity source.
  const summaryQuery = useQuery({
    queryKey: ["portfolio"],
    queryFn: () => getPortfolio("paper"),
    refetchInterval: connected ? false : 4000,
  });
  const ordersQuery = useQuery({
    queryKey: ["copy-orders"],
    queryFn: () => getCopyOrders("paper", 50),
    refetchInterval: connected ? false : 4000,
  });
  const history = useQuery({
    queryKey: ["portfolio-history"],
    queryFn: () => getPortfolioHistory("paper"),
    refetchInterval: 5000,
  });

  const revalue = useMutation({
    mutationFn: () => revaluePortfolio("paper"),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["portfolio"] });
      queryClient.invalidateQueries({ queryKey: ["portfolio-history"] });
    },
  });

  const close = useMutation({
    mutationFn: (positionId: number) => closePosition(positionId),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["portfolio"] });
      queryClient.invalidateQueries({ queryKey: ["portfolio-history"] });
      queryClient.invalidateQueries({ queryKey: ["copy-orders"] });
    },
  });

  const summary = frame?.portfolio ?? summaryQuery.data;
  const orders = frame?.copy_orders ?? ordersQuery.data?.items ?? [];

  const series =
    history.data && history.data.items.length > 0
      ? [
          {
            key: "equity",
            name: "Portfolio equity",
            color: SERIES_COLORS[0],
            points: history.data.items.map((s) => ({
              t: Date.parse(s.timestamp),
              equity: Number(s.total_value),
            })),
          },
        ]
      : [];

  return (
    <div>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-semibold">Portfolio</h1>
          <span
            className={`flex items-center gap-1.5 text-xs ${
              connected ? "text-emerald-400" : "text-slate-500"
            }`}
          >
            <span
              className={`inline-block h-2 w-2 rounded-full ${
                connected ? "bg-emerald-400" : "bg-slate-600"
              }`}
            />
            {connected ? "Live" : "Polling"}
          </span>
          <span className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-400">paper</span>
        </div>
        <button
          onClick={() => revalue.mutate()}
          disabled={revalue.isPending}
          className="rounded bg-sky-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-600 disabled:opacity-40"
        >
          {revalue.isPending ? "Revaluing…" : "Revalue now"}
        </button>
      </div>

      {summary && (
        <div className="mt-6">
          <SummaryTiles s={summary} />
        </div>
      )}

      <div className="mt-6 rounded border border-slate-800 bg-slate-950 p-4">
        <h2 className="text-sm font-medium text-slate-200">Equity curve</h2>
        <div className="mt-3">
          <EquityChart series={series} />
        </div>
      </div>

      <div className="mt-6 grid gap-6 xl:grid-cols-2">
        <div>
          <h2 className="mb-2 text-sm font-medium text-slate-200">Open positions</h2>
          {close.isError && (
            <p className="mb-2 text-xs text-red-400">
              Close failed: {(close.error as Error)?.message ?? "unknown error"}
            </p>
          )}
          <PositionsTable
            positions={summary?.positions ?? []}
            onClose={(id) => close.mutate(id)}
            closingId={close.isPending ? (close.variables ?? null) : null}
          />
        </div>
        <div>
          <h2 className="mb-2 text-sm font-medium text-slate-200">Recent copy orders</h2>
          <OrdersTable orders={orders} />
        </div>
      </div>
    </div>
  );
}
