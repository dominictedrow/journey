// Display helpers for backend values (Decimals arrive as strings; null means "not computed").

const DASH = "—";

function toNumber(value: string | number | null | undefined): number | null {
  if (value === null || value === undefined || value === "") return null;
  const n = typeof value === "number" ? value : Number(value);
  return Number.isFinite(n) ? n : null;
}

export function fmtNum(value: string | number | null | undefined, digits = 2): string {
  const n = toNumber(value);
  return n === null ? DASH : n.toLocaleString(undefined, { maximumFractionDigits: digits });
}

export function fmtUsd(value: string | number | null | undefined): string {
  const n = toNumber(value);
  if (n === null) return DASH;
  return `${n < 0 ? "-" : ""}$${Math.abs(n).toLocaleString(undefined, {
    maximumFractionDigits: 2,
  })}`;
}

export function fmtPct(value: string | number | null | undefined, digits = 1): string {
  const n = toNumber(value);
  return n === null ? DASH : `${(n * 100).toFixed(digits)}%`;
}

export function shortAddress(address: string): string {
  return `${address.slice(0, 6)}…${address.slice(-4)}`;
}

export function fmtDate(iso: string | null | undefined): string {
  if (!iso) return DASH;
  return new Date(iso).toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Tailwind text color for a signed PnL/ROI value. */
export function pnlClass(value: string | number | null | undefined): string {
  const n = toNumber(value);
  if (n === null || n === 0) return "text-slate-300";
  return n > 0 ? "text-emerald-400" : "text-red-400";
}
