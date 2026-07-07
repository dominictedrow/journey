import { Suspense } from "react";

import { WalletDetail } from "./WalletDetail";

// Wallet detail keys off the `?address=` query param (not a path segment) so the whole app
// is a single static bundle — the Tauri `.dmg` ships these files and talks to the remote
// backend at runtime. `useSearchParams` requires a Suspense boundary under static export.
export default function Page() {
  return (
    <Suspense fallback={<p className="text-slate-500">Loading wallet…</p>}>
      <WalletDetail />
    </Suspense>
  );
}
