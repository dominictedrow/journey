"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { getRiskStatus, setKillSwitch, type RiskEvent, type RiskLimits } from "@/lib/api";
import { fmtDate, fmtNum, fmtPct } from "@/lib/format";
import { useLiveState } from "@/lib/ws";

const LIMIT_ROWS: { key: keyof RiskLimits; label: string; kind: "usd" | "num" | "pct" }[] = [
  { key: "max_total_exposure_pct", label: "Max total exposure (% of equity)", kind: "pct" },
  { key: "max_position_size_pct", label: "Max position size (% of equity)", kind: "pct" },
  { key: "max_per_market_exposure_pct", label: "Max per-market (% of equity)", kind: "pct" },
  { key: "max_per_wallet_exposure_pct", label: "Max per-wallet (% of equity)", kind: "pct" },
  { key: "max_open_positions", label: "Max open positions", kind: "num" },
  { key: "max_daily_loss", label: "Max daily loss", kind: "usd" },
  { key: "max_weekly_loss", label: "Max weekly loss", kind: "usd" },
  { key: "max_monthly_loss", label: "Max monthly loss", kind: "usd" },
  { key: "min_market_liquidity", label: "Min market liquidity", kind: "usd" },
  { key: "max_slippage_pct", label: "Max slippage %", kind: "num" },
  { key: "max_copy_delay_seconds", label: "Max copy delay (s)", kind: "num" },
];

function severityClass(severity: string): string {
  if (severity === "critical") return "bg-red-900 text-red-300";
  if (severity === "warning") return "bg-amber-900 text-amber-300";
  return "bg-slate-800 text-slate-400";
}

function EventsTable({ events }: { events: RiskEvent[] }) {
  return (
    <div className="max-h-96 overflow-auto rounded border border-slate-800">
      <table className="w-full text-xs">
        <thead className="sticky top-0 bg-slate-900 text-left text-slate-400">
          <tr>
            <th className="px-2 py-1.5 font-medium">Time</th>
            <th className="px-2 py-1.5 font-medium">Type</th>
            <th className="px-2 py-1.5 font-medium">Severity</th>
            <th className="px-2 py-1.5 font-medium">Action</th>
            <th className="px-2 py-1.5 font-medium">Detail</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {events.map((e) => (
            <tr key={e.id} className="text-slate-300">
              <td className="px-2 py-1 text-slate-400">{fmtDate(e.timestamp)}</td>
              <td className="px-2 py-1 font-mono">{e.type}</td>
              <td className="px-2 py-1">
                <span className={`rounded px-2 py-0.5 ${severityClass(e.severity)}`}>
                  {e.severity}
                </span>
              </td>
              <td className="px-2 py-1 text-slate-400">{e.action_taken ?? "—"}</td>
              <td className="px-2 py-1 text-slate-500">
                {typeof e.context?.reason === "string" ? e.context.reason : ""}
              </td>
            </tr>
          ))}
          {events.length === 0 && (
            <tr>
              <td colSpan={5} className="px-2 py-6 text-center text-slate-500">
                No risk events yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

export default function RiskPage() {
  const queryClient = useQueryClient();
  const { frame, connected } = useLiveState("paper");

  const status = useQuery({
    queryKey: ["risk-status"],
    queryFn: getRiskStatus,
    refetchInterval: connected ? false : 4000,
  });

  const [reason, setReason] = useState("");
  const toggle = useMutation({
    mutationFn: (enabled: boolean) => setKillSwitch(enabled, reason || undefined),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["risk-status"] }),
  });

  const killSwitch = frame?.kill_switch ?? status.data?.kill_switch;
  const limits = status.data?.limits;
  const events = frame?.risk_events ?? status.data?.recent_events ?? [];
  const engaged = killSwitch?.engaged ?? false;

  return (
    <div>
      <div className="flex items-center gap-3">
        <h1 className="text-2xl font-semibold">Risk</h1>
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
        {status.data && (
          <span className="rounded bg-slate-800 px-2 py-0.5 text-xs text-slate-400">
            execution: {status.data.execution_mode}
          </span>
        )}
      </div>

      {/* Kill switch */}
      <div
        className={`mt-6 rounded border p-4 ${
          engaged ? "border-red-800 bg-red-950" : "border-slate-800 bg-slate-950"
        }`}
      >
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h2 className="text-sm font-medium text-slate-200">Kill switch</h2>
            <p className={`mt-1 text-sm ${engaged ? "text-red-300" : "text-slate-400"}`}>
              {engaged
                ? `ENGAGED — all copy execution halted${
                    killSwitch?.reason ? ` (${killSwitch.reason})` : ""
                  }`
                : "Clear — copy execution allowed"}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <input
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="reason (optional)"
              className="rounded border border-slate-700 bg-slate-900 px-2 py-1.5 text-sm text-slate-100"
            />
            <button
              onClick={() => toggle.mutate(!engaged)}
              disabled={toggle.isPending}
              className={`rounded px-4 py-1.5 text-sm font-medium text-white disabled:opacity-40 ${
                engaged ? "bg-emerald-700 hover:bg-emerald-600" : "bg-red-700 hover:bg-red-600"
              }`}
            >
              {engaged ? "Disengage" : "Engage kill switch"}
            </button>
          </div>
        </div>
      </div>

      <div className="mt-6 grid gap-6 lg:grid-cols-[1fr,20rem]">
        {/* Risk events */}
        <div>
          <h2 className="mb-2 text-sm font-medium text-slate-200">Risk events</h2>
          <EventsTable events={events} />
        </div>

        {/* Limits */}
        <div className="space-y-6">
          {limits && (
            <div className="rounded border border-slate-800 bg-slate-950 p-4">
              <h2 className="text-sm font-medium text-slate-200">Configured limits</h2>
              <table className="mt-2 w-full text-xs">
                <tbody className="divide-y divide-slate-800">
                  {LIMIT_ROWS.map(({ key, label, kind }) => (
                    <tr key={key} className="text-slate-400">
                      <td className="py-1 pr-2">{label}</td>
                      <td className="py-1 text-right font-mono text-slate-200">
                        {kind === "usd"
                          ? `$${fmtNum(limits[key] as string)}`
                          : kind === "pct"
                            ? fmtPct(limits[key] as string)
                            : fmtNum(limits[key] as number)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-2 text-xs text-slate-600">
                Set in backend config; edit env to change.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
