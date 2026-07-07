"use client";

import { keepPreviousData, useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";

import { getWallets, updateWallet, type WalletSortKey } from "@/lib/api";
import { fmtDate, fmtNum, fmtPct, fmtUsd, pnlClass, shortAddress } from "@/lib/format";

const PAGE_SIZE = 25;

const COLUMNS: { key: WalletSortKey; label: string }[] = [
  { key: "total_volume", label: "Volume" },
  { key: "total_trades", label: "Trades" },
  { key: "roi_lifetime", label: "ROI" },
  { key: "realized_pnl", label: "Realized PnL" },
  { key: "win_rate", label: "Win rate" },
  { key: "sharpe", label: "Sharpe" },
  { key: "last_active", label: "Last active" },
];

export default function WalletsPage() {
  const [search, setSearch] = useState("");
  const [trackedOnly, setTrackedOnly] = useState(false);
  const [sort, setSort] = useState<WalletSortKey>("total_volume");
  const [order, setOrder] = useState<"asc" | "desc">("desc");
  const [page, setPage] = useState(0);
  const queryClient = useQueryClient();

  const params = {
    search: search || undefined,
    tracked: trackedOnly || undefined,
    sort,
    order,
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
  };
  const { data, isPending, error } = useQuery({
    queryKey: ["wallets", params],
    queryFn: () => getWallets(params),
    placeholderData: keepPreviousData,
  });

  const track = useMutation({
    mutationFn: ({ address, tracked }: { address: string; tracked: boolean }) =>
      updateWallet(address, { is_tracked: tracked }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["wallets"] }),
  });

  const sortBy = (key: WalletSortKey) => {
    if (key === sort) {
      setOrder(order === "desc" ? "asc" : "desc");
    } else {
      setSort(key);
      setOrder("desc");
    }
    setPage(0);
  };

  const pageCount = data ? Math.max(Math.ceil(data.total / PAGE_SIZE), 1) : 1;

  return (
    <div>
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Wallets</h1>
        {data && <p className="text-sm text-slate-500">{data.total} wallets</p>}
      </div>

      <div className="mt-6 flex flex-wrap items-center gap-3">
        <input
          type="search"
          value={search}
          onChange={(e) => {
            setSearch(e.target.value);
            setPage(0);
          }}
          placeholder="Search address…"
          className="w-72 rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 focus:border-slate-500 focus:outline-none"
        />
        <label className="flex items-center gap-2 text-sm text-slate-300">
          <input
            type="checkbox"
            checked={trackedOnly}
            onChange={(e) => {
              setTrackedOnly(e.target.checked);
              setPage(0);
            }}
          />
          Tracked only
        </label>
      </div>

      {error && (
        <p className="mt-6 rounded border border-red-900 bg-red-950 p-3 text-sm text-red-300">
          Failed to load wallets: {error.message}
        </p>
      )}

      <div className="mt-4 overflow-x-auto rounded border border-slate-800">
        <table className="w-full text-sm">
          <thead className="bg-slate-900 text-left text-slate-400">
            <tr>
              <th className="px-3 py-2 font-medium">Address</th>
              {COLUMNS.map(({ key, label }) => (
                <th key={key} className="px-3 py-2 font-medium">
                  <button
                    onClick={() => sortBy(key)}
                    className={`hover:text-slate-200 ${sort === key ? "text-slate-100" : ""}`}
                  >
                    {label}
                    {sort === key ? (order === "desc" ? " ↓" : " ↑") : ""}
                  </button>
                </th>
              ))}
              <th className="px-3 py-2 font-medium">Source</th>
              <th className="px-3 py-2 font-medium">Tracked</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800">
            {data?.items.map((w) => (
              <tr key={w.address} className="hover:bg-slate-900/50">
                <td className="px-3 py-2 font-mono">
                  <Link
                    href={`/wallets/detail?address=${w.address}`}
                    className="text-sky-400 hover:underline"
                  >
                    {shortAddress(w.address)}
                  </Link>
                </td>
                <td className="px-3 py-2">{fmtUsd(w.total_volume)}</td>
                <td className="px-3 py-2">{fmtNum(w.total_trades, 0)}</td>
                <td className={`px-3 py-2 ${pnlClass(w.metrics?.roi_lifetime)}`}>
                  {fmtPct(w.metrics?.roi_lifetime)}
                </td>
                <td className={`px-3 py-2 ${pnlClass(w.metrics?.realized_pnl)}`}>
                  {fmtUsd(w.metrics?.realized_pnl)}
                </td>
                <td className="px-3 py-2">{fmtPct(w.metrics?.win_rate)}</td>
                <td className="px-3 py-2">{fmtNum(w.metrics?.sharpe)}</td>
                <td className="px-3 py-2 text-slate-400">{fmtDate(w.last_active)}</td>
                <td className="px-3 py-2 text-slate-400">{w.discovery_source ?? "—"}</td>
                <td className="px-3 py-2">
                  <button
                    onClick={() => track.mutate({ address: w.address, tracked: !w.is_tracked })}
                    className={`rounded px-2 py-1 text-xs font-medium ${
                      w.is_tracked
                        ? "bg-emerald-900 text-emerald-300 hover:bg-emerald-800"
                        : "bg-slate-800 text-slate-400 hover:bg-slate-700"
                    }`}
                  >
                    {w.is_tracked ? "Tracked" : "Track"}
                  </button>
                </td>
              </tr>
            ))}
            {!isPending && data?.items.length === 0 && (
              <tr>
                <td colSpan={10} className="px-3 py-8 text-center text-slate-500">
                  No wallets yet — run discovery or backfill a wallet to get started.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="mt-4 flex items-center gap-3 text-sm text-slate-400">
        <button
          onClick={() => setPage((p) => Math.max(p - 1, 0))}
          disabled={page === 0}
          className="rounded bg-slate-800 px-3 py-1 hover:bg-slate-700 disabled:opacity-40"
        >
          Prev
        </button>
        <span>
          Page {page + 1} of {pageCount}
        </span>
        <button
          onClick={() => setPage((p) => p + 1)}
          disabled={page + 1 >= pageCount}
          className="rounded bg-slate-800 px-3 py-1 hover:bg-slate-700 disabled:opacity-40"
        >
          Next
        </button>
      </div>
    </div>
  );
}
