"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";

import {
  ApiError,
  getActiveStrategy,
  getStrategies,
  getStrategy,
  runOptimizer,
  type CandidateSummary,
  type OptimizerDecision,
  type Strategy,
  type StrategyDetail,
  type StrategySpec,
} from "@/lib/api";
import { fmtDate, fmtNum, fmtPct } from "@/lib/format";

const DECISION_STYLE: Record<OptimizerDecision, { label: string; cls: string }> = {
  adopted: { label: "adopted", cls: "bg-emerald-900 text-emerald-300" },
  kept_active: { label: "kept active", cls: "bg-slate-800 text-slate-300" },
  no_candidate: { label: "no candidate", cls: "bg-amber-900 text-amber-300" },
};

function specLine(spec: StrategySpec): string {
  return `Top-${spec.top_n} · ${spec.category} · ${spec.weighting}-weighted · ${spec.sizing_mode} sizing`;
}

function objectiveLabel(name: string): string {
  return name.replace(/_/g, " ");
}

/** Format an objective value the way its metric reads (returns as %, ratios as numbers). */
function fmtObjective(name: string, value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  return name === "total_return" || name === "cagr" ? fmtPct(value) : fmtNum(value, 3);
}

function StrategyCard({
  title,
  accent,
  strategy,
}: {
  title: string;
  accent: string;
  strategy: StrategyDetail | null;
}) {
  return (
    <div className="rounded border border-slate-800 bg-slate-950 p-4">
      <div className="flex items-center justify-between">
        <h2 className={`text-sm font-medium ${accent}`}>{title}</h2>
        {strategy && <span className="font-mono text-xs text-slate-500">#{strategy.id}</span>}
      </div>
      {!strategy ? (
        <p className="mt-3 text-sm text-slate-500">None yet — run the optimizer.</p>
      ) : (
        <>
          <p className="mt-2 text-sm text-slate-200">{specLine(strategy.params)}</p>
          <dl className="mt-3 grid grid-cols-2 gap-y-1 text-xs text-slate-400">
            <dt>Adopted</dt>
            <dd className="text-right text-slate-300">{fmtDate(strategy.adopted_at)}</dd>
            {strategy.rationale && (
              <>
                <dt>{objectiveLabel(strategy.rationale.objective)}</dt>
                <dd className="text-right font-mono text-slate-300">
                  {fmtObjective(
                    strategy.rationale.objective,
                    strategy.rationale.chosen?.objective_value,
                  )}
                </dd>
                <dt>Max drawdown</dt>
                <dd className="text-right font-mono text-slate-300">
                  {fmtPct(strategy.rationale.chosen?.max_drawdown)}
                </dd>
              </>
            )}
          </dl>
        </>
      )}
    </div>
  );
}

function CandidatesTable({
  candidates,
  objective,
  chosenName,
}: {
  candidates: CandidateSummary[];
  objective: string;
  chosenName: string | null;
}) {
  const ranked = [...candidates].sort(
    (a, b) => (b.objective_value ?? -Infinity) - (a.objective_value ?? -Infinity),
  );
  return (
    <div className="overflow-x-auto rounded border border-slate-800">
      <table className="w-full text-xs">
        <thead className="bg-slate-900 text-left text-slate-400">
          <tr>
            <th className="px-3 py-2 font-medium">Candidate</th>
            <th className="px-3 py-2 font-medium">Wallets</th>
            <th className="px-3 py-2 font-medium">{objectiveLabel(objective)}</th>
            <th className="px-3 py-2 font-medium">Return</th>
            <th className="px-3 py-2 font-medium">Sharpe</th>
            <th className="px-3 py-2 font-medium">Max DD</th>
            <th className="px-3 py-2 font-medium">Fills</th>
            <th className="px-3 py-2 font-medium">Risk</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800">
          {ranked.map((c) => (
            <tr
              key={c.name}
              className={
                c.name === chosenName ? "bg-emerald-950/40 text-slate-200" : "text-slate-300"
              }
            >
              <td className="px-3 py-1.5">
                {c.name === chosenName && <span className="mr-1 text-emerald-400">★</span>}
                {c.name}
              </td>
              <td className="px-3 py-1.5 text-slate-400">{c.wallets}</td>
              <td className="px-3 py-1.5 font-mono">
                {fmtObjective(objective, c.objective_value)}
              </td>
              <td className="px-3 py-1.5 font-mono">{fmtPct(c.total_return)}</td>
              <td className="px-3 py-1.5 font-mono">{fmtNum(c.sharpe, 2)}</td>
              <td className="px-3 py-1.5 font-mono">{fmtPct(c.max_drawdown)}</td>
              <td className="px-3 py-1.5 text-slate-400">{c.trades_executed ?? "—"}</td>
              <td className="px-3 py-1.5">
                <span className={c.passes_risk ? "text-emerald-400" : "text-red-400"}>
                  {c.passes_risk ? "pass" : "fail"}
                </span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function OptimizationPage() {
  const queryClient = useQueryClient();
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [queued, setQueued] = useState(false);
  const baseline = useRef<number | null>(null);

  const active = useQuery({
    queryKey: ["strategy-active"],
    queryFn: getActiveStrategy,
    retry: (count, error) => !(error instanceof ApiError && error.status === 404) && count < 2,
  });
  const activeStrategy =
    active.data ??
    (active.error instanceof ApiError && active.error.status === 404 ? null : undefined);

  const list = useQuery({
    queryKey: ["strategies"],
    queryFn: () => getStrategies(50),
    refetchInterval: queued ? 3000 : false,
  });

  // Clear the "queued" polling once a new run has landed a strategy row.
  useEffect(() => {
    if (queued && list.data && baseline.current !== null && list.data.total > baseline.current) {
      setQueued(false);
      queryClient.invalidateQueries({ queryKey: ["strategy-active"] });
    }
  }, [queued, list.data, queryClient]);

  const detail = useQuery({
    queryKey: ["strategy", selectedId],
    queryFn: () => getStrategy(selectedId as number),
    enabled: selectedId !== null,
  });

  const optimize = useMutation({
    mutationFn: runOptimizer,
    onSuccess: () => {
      baseline.current = list.data?.total ?? 0;
      setQueued(true);
    },
  });

  // The most recent run's recommendation (top of the history list).
  const latest = list.data?.items[0] ?? null;
  const latestDetail = useQuery({
    queryKey: ["strategy", latest?.id],
    queryFn: () => getStrategy(latest?.id as number),
    enabled: latest !== null,
  });
  const recommendation = latestDetail.data ?? null;

  const shown = detail.data ?? recommendation;

  return (
    <div>
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Optimization</h1>
        <div className="flex items-center gap-3">
          {queued && <span className="text-xs text-amber-400">optimizing…</span>}
          <button
            onClick={() => optimize.mutate()}
            disabled={optimize.isPending || queued}
            className="rounded bg-sky-700 px-3 py-2 text-sm font-medium text-white hover:bg-sky-600 disabled:opacity-40"
          >
            {optimize.isPending || queued ? "Running…" : "Run optimizer"}
          </button>
        </div>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Backtests a grid of candidate strategies each week and adopts the best only when it beats
        the active strategy by the configured margin and clears the risk constraints.
      </p>
      {optimize.error && (
        <p className="mt-2 text-xs text-red-400">{(optimize.error as Error).message}</p>
      )}

      {/* Current vs recommended */}
      <div className="mt-6 grid gap-6 md:grid-cols-2">
        <StrategyCard
          title="Active strategy"
          accent="text-emerald-300"
          strategy={activeStrategy ?? null}
        />
        <StrategyCard
          title="Latest recommendation"
          accent="text-sky-300"
          strategy={recommendation}
        />
      </div>

      {/* History */}
      <div className="mt-6 overflow-x-auto rounded border border-slate-800">
        <table className="w-full text-sm">
          <thead className="bg-slate-900 text-left text-slate-400">
            <tr>
              <th className="px-3 py-2 font-medium">Run</th>
              <th className="px-3 py-2 font-medium">Strategy</th>
              <th className="px-3 py-2 font-medium">Decision</th>
              <th className="px-3 py-2 font-medium">Active</th>
              <th className="px-3 py-2 font-medium">When</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {list.data?.items.map((s: Strategy) => (
              <tr
                key={s.id}
                onClick={() => setSelectedId(s.id)}
                className={`cursor-pointer hover:bg-slate-900/50 ${
                  (shown?.id ?? -1) === s.id ? "bg-slate-900/70" : ""
                }`}
              >
                <td className="px-3 py-2 font-mono text-sky-400">#{s.id}</td>
                <td className="px-3 py-2 text-slate-300">{specLine(s.params)}</td>
                <td className="px-3 py-2 text-slate-400">{s.name}</td>
                <td className="px-3 py-2">
                  {s.is_active && (
                    <span className="rounded bg-emerald-900 px-2 py-0.5 text-xs text-emerald-300">
                      active
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-slate-400">{fmtDate(s.created_at)}</td>
              </tr>
            ))}
            {!list.isPending && list.data?.items.length === 0 && (
              <tr>
                <td colSpan={5} className="px-3 py-8 text-center text-slate-500">
                  No optimizer runs yet — click “Run optimizer”.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {/* Rationale of the selected (or latest) run */}
      {shown?.rationale && (
        <div className="mt-6 rounded border border-slate-800 bg-slate-950 p-4">
          <div className="flex items-center justify-between">
            <h2 className="text-sm font-medium text-slate-200">Run #{shown.id} rationale</h2>
            <span
              className={`rounded px-2 py-0.5 text-xs font-medium ${
                DECISION_STYLE[shown.rationale.decision].cls
              }`}
            >
              {DECISION_STYLE[shown.rationale.decision].label}
            </span>
          </div>
          <p className="mt-2 text-sm text-slate-300">{shown.rationale.summary}</p>
          <dl className="mt-3 flex flex-wrap gap-x-6 gap-y-1 text-xs text-slate-400">
            <div>
              Objective:{" "}
              <span className="text-slate-300">{objectiveLabel(shown.rationale.objective)}</span>
            </div>
            <div>
              Min improvement:{" "}
              <span className="text-slate-300">{fmtPct(shown.rationale.min_improvement)}</span>
            </div>
            <div>
              Max drawdown cap:{" "}
              <span className="text-slate-300">{fmtPct(shown.rationale.max_drawdown_cap)}</span>
            </div>
            <div>
              Lookback: <span className="text-slate-300">{shown.rationale.lookback_days}d</span>
            </div>
          </dl>

          {shown.rationale.candidates.length > 0 && (
            <div className="mt-4">
              <h3 className="mb-2 text-xs font-medium text-slate-400">
                Candidates ({shown.rationale.candidates.length}, best first)
              </h3>
              <CandidatesTable
                candidates={shown.rationale.candidates}
                objective={shown.rationale.objective}
                chosenName={shown.rationale.chosen?.name ?? null}
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
