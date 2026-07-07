"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";

import { getDiscoveryStatus, getWallets, runDiscovery } from "@/lib/api";
import { fmtDate, fmtUsd, shortAddress } from "@/lib/format";

function Stat({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="rounded border border-slate-800 bg-slate-900/50 p-4">
      <div className="text-xs text-slate-500">{label}</div>
      <div className="mt-1 text-2xl font-semibold text-slate-100">{value}</div>
    </div>
  );
}

export default function DiscoveryPage() {
  const queryClient = useQueryClient();
  const [taskId, setTaskId] = useState<string | null>(null);

  const { data: status, error } = useQuery({
    queryKey: ["discovery"],
    queryFn: getDiscoveryStatus,
    refetchInterval: 30_000,
  });
  const { data: recent } = useQuery({
    queryKey: ["wallets", "recently-discovered"],
    queryFn: () => getWallets({ sort: "discovered_at", order: "desc", limit: 10 }),
    refetchInterval: 30_000,
  });

  const sweep = useMutation({
    mutationFn: runDiscovery,
    onSuccess: (res) => {
      setTaskId(res.task_id);
      // Give the worker a moment, then pull fresh counts.
      setTimeout(() => {
        queryClient.invalidateQueries({ queryKey: ["discovery"] });
        queryClient.invalidateQueries({ queryKey: ["wallets"] });
      }, 5000);
    },
  });

  return (
    <div>
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Discovery</h1>
        <button
          onClick={() => sweep.mutate()}
          disabled={sweep.isPending}
          className="rounded bg-emerald-700 px-4 py-2 text-sm font-medium hover:bg-emerald-600 disabled:opacity-50"
        >
          {sweep.isPending ? "Queuing…" : "Run discovery sweep"}
        </button>
      </div>
      <p className="mt-1 text-sm text-slate-500">
        Sources: Polymarket trader leaderboard (PnL & volume, monthly & lifetime) and top holders of
        the highest-volume active markets. Also runs automatically every 6 hours.
      </p>

      {taskId && (
        <p className="mt-3 rounded border border-sky-900 bg-sky-950 p-2 text-sm text-sky-300">
          Sweep queued (task {taskId}) — counts refresh as the worker reports in.
        </p>
      )}
      {sweep.error && (
        <p className="mt-3 rounded border border-red-900 bg-red-950 p-2 text-sm text-red-300">
          Failed to queue sweep: {sweep.error.message}
        </p>
      )}
      {error && (
        <p className="mt-3 rounded border border-red-900 bg-red-950 p-2 text-sm text-red-300">
          Failed to load discovery status: {error.message}
        </p>
      )}

      {status && (
        <div className="mt-6 grid grid-cols-1 gap-4 md:grid-cols-3">
          <Stat label="Candidate wallets" value={status.total_wallets} />
          <Stat label="Tracked wallets" value={status.tracked_wallets} />
          <Stat label="Last discovery" value={fmtDate(status.last_discovered_at)} />
        </div>
      )}

      <div className="mt-8 grid grid-cols-1 gap-8 lg:grid-cols-2">
        <section>
          <h2 className="text-lg font-semibold">Candidates by source</h2>
          <div className="mt-3 overflow-x-auto rounded border border-slate-800">
            <table className="w-full text-sm">
              <thead className="bg-slate-900 text-left text-slate-400">
                <tr>
                  <th className="px-3 py-2 font-medium">Source</th>
                  <th className="px-3 py-2 font-medium">Wallets</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {status?.by_source.map((row) => (
                  <tr key={row.source ?? "chain"}>
                    <td className="px-3 py-2">{row.source ?? "seen on-chain (no attribution)"}</td>
                    <td className="px-3 py-2">{row.count}</td>
                  </tr>
                ))}
                {status && status.by_source.length === 0 && (
                  <tr>
                    <td colSpan={2} className="px-3 py-6 text-center text-slate-500">
                      No candidates yet — run a sweep.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>

        <section>
          <h2 className="text-lg font-semibold">Recently discovered</h2>
          <div className="mt-3 overflow-x-auto rounded border border-slate-800">
            <table className="w-full text-sm">
              <thead className="bg-slate-900 text-left text-slate-400">
                <tr>
                  <th className="px-3 py-2 font-medium">Address</th>
                  <th className="px-3 py-2 font-medium">Source</th>
                  <th className="px-3 py-2 font-medium">Volume</th>
                  <th className="px-3 py-2 font-medium">Discovered</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {recent?.items.map((w) => (
                  <tr key={w.address} className="hover:bg-slate-900/50">
                    <td className="px-3 py-2 font-mono">
                      <Link
                        href={`/wallets/detail?address=${w.address}`}
                        className="text-sky-400 hover:underline"
                      >
                        {shortAddress(w.address)}
                      </Link>
                    </td>
                    <td className="px-3 py-2 text-slate-400">{w.discovery_source ?? "—"}</td>
                    <td className="px-3 py-2">{fmtUsd(w.total_volume)}</td>
                    <td className="px-3 py-2 text-slate-400">{fmtDate(w.discovered_at)}</td>
                  </tr>
                ))}
                {recent && recent.items.length === 0 && (
                  <tr>
                    <td colSpan={4} className="px-3 py-6 text-center text-slate-500">
                      Nothing discovered yet.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </section>
      </div>
    </div>
  );
}
