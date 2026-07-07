"use client";

import { useRouter } from "next/navigation";
import { useEffect } from "react";

// Client-side redirect (not server `redirect()`): in the Tauri static bundle there is no server
// to issue a 3xx, so the landing route navigates on hydration. Works identically under the
// Docker SSR build.
export default function Home() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/portfolio");
  }, [router]);
  return <p className="p-8 text-slate-500">Loading…</p>;
}
