"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { Fragment, useState } from "react";

import {
  getRankings,
  rebuildRankings,
  type RankingCategory,
  type ScoreExplanationComponent,
  type WalletScoreDetail,
} from "@/lib/api";
import { fmtDate, fmtNum, shortAddress } from "@/lib/format";

const CATEGORIES: { key: RankingCategory; label: string }[] = [
  { key: "overall", label: "Overall" },
  { key: "politics", label: "Politics" },
  { key: "crypto", label: "Crypto" },
  { key: "sports", label: "Sports" },
  { key: "economics", label: "Economics" },
  { key: "hft", label: "HFT" },
  { key: "long_term", label: "Long-term" },
  { key: "early_info", label: "Early info" },
  { key: "low_risk", label: "Low risk" },
];

const COMPONENTS: { key: string; label: string }[] = [
  { key: "historical", label: "Historical" },
  { key: "recent", label: "Recent" },
  { key: "copy_quality", label: "Copy quality" },
  { key: "risk", label: "Risk" },
];

function scorePct(value: string | number | null | undefined): number {
  const n = typeof value === "number" ? value : Number(value ?? NaN);
  return Number.isFinite(n) ? Math.min(Math.max(n, 0), 1) * 100 : 0;
}

function ScoreBar({
  value,
  className = "",
}: {
  value: string | number | null;
  className?: string;
}) {
  return (
    <div className={`h-1.5 w-full rounded bg-slate-800 ${className}`}>
      <div className="h-1.5 rounded bg-sky-500" style={{ width: `${scorePct(value)}%` }} />
    </div>
  );
}

function componentScore(detail: WalletScoreDetail, key: string): string | null {
  switch (key) {
    case "historical":
      return detail.historical_component;
    case "recent":
      return detail.recent_component;
    case "copy_quality":
      return detail.copy_quality_component;
    default:
      return detail.risk_component;
  }
}

function ExplanationPanel({ detail }: { detail: WalletScoreDetail }) {
  const components = detail.explanation?.components;
  if (!components) return <p className="text-sm text-slate-500">No explanation recorded.</p>;
  return (
    <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
      {COMPONENTS.map(({ key, label }) => {
        const comp: ScoreExplanationComponent | undefined = components[key];
        if (!comp) return null;
        return (
          <div key={key} className="rounded border border-slate-800 bg-slate-950 p-3">
            <div className="flex items-baseline justify-between">
              <span className="text-sm font-medium text-slate-200">{label}</span>
              <span className="text-xs text-slate-500">
                weight {comp.weight} · +{fmtNum(comp.contribution, 3)}
              </span>
            </div>
            <div className="mt-2 flex items-center gap-2">
              <ScoreBar value={comp.score} />
              <span className="w-12 text-right text-sm text-slate-300">
                {fmtNum(comp.score, 3)}
              </span>
            </div>
            {comp.neutral_prior ? (
              <p className="mt-2 text-xs text-slate-500">
                Neutral prior — no data for this component yet.
              </p>
            ) : (
              <table className="mt-2 w-full text-xs">
                <tbody>
                  {Object.entries(comp.inputs).map(([metric, input]) => (
                    <tr key={metric} className="text-slate-400">
                      <td className="py-0.5 pr-2">{metric.replace(/_/g, " ")}</td>
                      <td className="py-0.5 pr-2 text-right font-mono">{fmtNum(input.value, 4)}</td>
                      <td className="py-0.5 text-right font-mono text-slate-300">
                        {fmtNum(input.score, 3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        );
      })}
    </div>
  );
}

export default function RankingsPage() {
  const [category, setCategory] = useState<RankingCategory>("overall");
  const [expanded, setExpanded] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const { data, isPending, error } = useQuery({
    queryKey: ["rankings", category],
    queryFn: () => getRankings(category),
  });

  const rebuild = useMutation({
    mutationFn: rebuildRankings,
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["rankings"] }),
  });

  const counts = new Map(data?.categories.map((c) => [c.category, c.count]));

  return (
    <div>
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Rankings</h1>
        <div className="flex items-center gap-3">
          {data?.computed_at && (
            <span className="text-sm text-slate-500">Generation: {fmtDate(data.computed_at)}</span>
          )}
          <button
            onClick={() => rebuild.mutate()}
            disabled={rebuild.isPending}
            className="rounded bg-sky-700 px-3 py-1.5 text-sm font-medium text-white hover:bg-sky-600 disabled:opacity-40"
          >
            {rebuild.isPending ? "Rebuilding…" : "Rebuild now"}
          </button>
        </div>
      </div>

      <div className="mt-6 flex flex-wrap gap-2">
        {CATEGORIES.map(({ key, label }) => (
          <button
            key={key}
            onClick={() => {
              setCategory(key);
              setExpanded(null);
            }}
            className={`rounded-full px-3 py-1 text-sm ${
              category === key
                ? "bg-sky-800 text-white"
                : "bg-slate-800 text-slate-400 hover:bg-slate-700 hover:text-slate-200"
            }`}
          >
            {label}
            {counts.has(key) ? ` (${counts.get(key)})` : ""}
          </button>
        ))}
      </div>

      {error && (
        <p className="mt-6 rounded border border-red-900 bg-red-950 p-3 text-sm text-red-300">
          Failed to load rankings: {error.message}
        </p>
      )}
      {rebuild.error && (
        <p className="mt-6 rounded border border-red-900 bg-red-950 p-3 text-sm text-red-300">
          Rebuild failed: {rebuild.error.message}
        </p>
      )}

      <div className="mt-4 overflow-x-auto rounded border border-slate-800">
        <table className="w-full text-sm">
          <thead className="bg-slate-900 text-left text-slate-400">
            <tr>
              <th className="w-12 px-3 py-2 font-medium">#</th>
              <th className="px-3 py-2 font-medium">Wallet</th>
              <th className="w-56 px-3 py-2 font-medium">Score</th>
              {COMPONENTS.map(({ key, label }) => (
                <th key={key} className="px-3 py-2 font-medium">
                  {label}
                </th>
              ))}
              <th className="px-3 py-2 font-medium">Tracked</th>
              <th className="w-24 px-3 py-2" />
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {data?.items.map((entry) => (
              <Fragment key={entry.wallet.address}>
                <tr className="hover:bg-slate-900/50">
                  <td className="px-3 py-2 text-slate-400">{entry.rank}</td>
                  <td className="px-3 py-2 font-mono">
                    <Link
                      href={`/wallets/detail?address=${entry.wallet.address}`}
                      className="text-sky-400 hover:underline"
                    >
                      {shortAddress(entry.wallet.address)}
                    </Link>
                  </td>
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <ScoreBar value={entry.score} />
                      <span className="w-14 text-right font-mono text-slate-200">
                        {fmtNum(entry.score, 3)}
                      </span>
                    </div>
                  </td>
                  {COMPONENTS.map(({ key }) => (
                    <td key={key} className="px-3 py-2 text-slate-300">
                      {entry.score_detail
                        ? fmtNum(componentScore(entry.score_detail, key), 3)
                        : "—"}
                    </td>
                  ))}
                  <td className="px-3 py-2 text-slate-400">
                    {entry.wallet.is_tracked ? "Yes" : "No"}
                  </td>
                  <td className="px-3 py-2 text-right">
                    {entry.score_detail && (
                      <button
                        onClick={() =>
                          setExpanded(
                            expanded === entry.wallet.address ? null : entry.wallet.address,
                          )
                        }
                        className="text-xs text-slate-400 hover:text-slate-200"
                      >
                        {expanded === entry.wallet.address ? "Hide" : "Why?"}
                      </button>
                    )}
                  </td>
                </tr>
                {expanded === entry.wallet.address && entry.score_detail && (
                  <tr className="bg-slate-900/30">
                    <td colSpan={9} className="px-3 py-3">
                      <ExplanationPanel detail={entry.score_detail} />
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
            {!isPending && data?.items.length === 0 && (
              <tr>
                <td colSpan={9} className="px-3 py-8 text-center text-slate-500">
                  {data.computed_at
                    ? "No wallets qualify for this category yet."
                    : "No rankings yet — compute wallet metrics, then hit Rebuild now."}
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
