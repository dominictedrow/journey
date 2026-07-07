"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { fmtUsd } from "@/lib/format";

// Categorical slots validated for the slate-950 surface (dataviz six-checks:
// lightness band, chroma, CVD ΔE ≥ 12 adjacent, ≥ 3:1 contrast). Fixed order,
// never cycled — a run keeps its slot while selected.
export const SERIES_COLORS = ["#3987e5", "#199e70", "#c98500", "#9085e9"];
export const MAX_COMPARE = SERIES_COLORS.length;

export interface EquitySeries {
  key: string;
  name: string;
  color: string;
  points: { t: number; equity: number }[];
}

function fmtTick(ms: number): string {
  return new Date(ms).toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function fmtTooltipTime(ms: number): string {
  return new Date(ms).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

interface TooltipPayloadEntry {
  name?: string | number;
  value?: number | string;
  color?: string;
  payload?: { t: number };
}

function EquityTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayloadEntry[] }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="rounded border border-slate-700 bg-slate-900 px-3 py-2 text-xs shadow-lg">
      <p className="text-slate-400">{fmtTooltipTime(payload[0].payload?.t ?? 0)}</p>
      {payload.map((entry) => (
        <p key={String(entry.name)} className="mt-1 flex items-center gap-2 text-slate-200">
          <span
            className="inline-block h-2 w-2 rounded-full"
            style={{ backgroundColor: entry.color }}
          />
          {entry.name}: {fmtUsd(entry.value as number)}
        </p>
      ))}
    </div>
  );
}

export function EquityChart({ series }: { series: EquitySeries[] }) {
  if (series.length === 0 || series.every((s) => s.points.length === 0)) {
    return (
      <div className="flex h-64 items-center justify-center text-sm text-slate-500">
        No equity data.
      </div>
    );
  }
  return (
    <div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart margin={{ top: 8, right: 16, bottom: 4, left: 8 }}>
          <CartesianGrid stroke="#1e293b" strokeWidth={1} vertical={false} />
          <XAxis
            dataKey="t"
            type="number"
            domain={["dataMin", "dataMax"]}
            tickFormatter={fmtTick}
            tick={{ fill: "#64748b", fontSize: 11 }}
            tickLine={false}
            axisLine={{ stroke: "#334155" }}
            allowDuplicatedCategory={false}
          />
          <YAxis
            tickFormatter={(v: number) => fmtUsd(v)}
            tick={{ fill: "#64748b", fontSize: 11 }}
            tickLine={false}
            axisLine={false}
            width={80}
            domain={["auto", "auto"]}
          />
          <Tooltip content={<EquityTooltip />} cursor={{ stroke: "#475569", strokeWidth: 1 }} />
          {series.map((s) => (
            <Line
              key={s.key}
              data={s.points}
              dataKey="equity"
              name={s.name}
              stroke={s.color}
              strokeWidth={2}
              dot={false}
              activeDot={{ r: 4, strokeWidth: 0 }}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
      {series.length >= 2 && (
        <div className="mt-2 flex flex-wrap gap-4 px-2">
          {series.map((s) => (
            <span key={s.key} className="flex items-center gap-2 text-xs text-slate-300">
              <span className="inline-block h-0.5 w-4" style={{ backgroundColor: s.color }} />
              {s.name}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
