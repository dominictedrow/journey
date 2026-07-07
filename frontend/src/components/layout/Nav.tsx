"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const links = [
  { href: "/portfolio", label: "Portfolio" },
  { href: "/wallets", label: "Wallets" },
  { href: "/rankings", label: "Rankings" },
  { href: "/discovery", label: "Discovery" },
  { href: "/risk", label: "Risk" },
  { href: "/backtesting", label: "Backtesting" },
  { href: "/optimization", label: "Optimization" },
  { href: "/settings", label: "Settings" },
];

export function Nav() {
  const pathname = usePathname();
  return (
    <nav className="flex w-56 shrink-0 flex-col gap-1 border-r border-slate-800 bg-slate-950 p-4">
      <div className="mb-6 px-2 text-lg font-semibold text-slate-100">Copytrader</div>
      {links.map(({ href, label }) => (
        <Link
          key={href}
          href={href}
          className={`rounded px-3 py-2 text-sm transition-colors ${
            pathname?.startsWith(href)
              ? "bg-slate-800 text-white"
              : "text-slate-400 hover:bg-slate-900 hover:text-slate-200"
          }`}
        >
          {label}
        </Link>
      ))}
    </nav>
  );
}
