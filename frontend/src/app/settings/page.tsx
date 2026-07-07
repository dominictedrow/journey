"use client";

import { useState } from "react";

import { getHealth, type HealthResponse } from "@/lib/api";
import { useSettings } from "@/stores/settings";

type TestState =
  | { kind: "idle" }
  | { kind: "testing" }
  | { kind: "ok"; health: HealthResponse }
  | { kind: "error"; message: string };

export default function SettingsPage() {
  const backendUrl = useSettings((s) => s.backendUrl);
  const setBackendUrl = useSettings((s) => s.setBackendUrl);
  const [draft, setDraft] = useState<string | null>(null);
  const [test, setTest] = useState<TestState>({ kind: "idle" });

  const value = draft ?? backendUrl;

  const save = () => {
    setBackendUrl(value);
    setDraft(null);
    setTest({ kind: "idle" });
  };

  const testConnection = async () => {
    setBackendUrl(value);
    setDraft(null);
    setTest({ kind: "testing" });
    try {
      const health = await getHealth();
      setTest({ kind: "ok", health });
    } catch (err) {
      setTest({ kind: "error", message: err instanceof Error ? err.message : String(err) });
    }
  };

  return (
    <div className="max-w-xl">
      <h1 className="text-2xl font-semibold">Settings</h1>

      <div className="mt-8">
        <label htmlFor="backend-url" className="block text-sm font-medium text-slate-300">
          Backend URL
        </label>
        <p className="mt-1 text-sm text-slate-500">
          Point the dashboard at any Copytrader backend (Windows test box, Beelink, …).
        </p>
        <div className="mt-3 flex gap-2">
          <input
            id="backend-url"
            type="url"
            value={value}
            onChange={(e) => setDraft(e.target.value)}
            className="flex-1 rounded border border-slate-700 bg-slate-950 px-3 py-2 text-sm text-slate-100 focus:border-slate-500 focus:outline-none"
            placeholder="http://192.168.1.50:8000"
          />
          <button
            onClick={save}
            className="rounded bg-slate-700 px-4 py-2 text-sm font-medium hover:bg-slate-600"
          >
            Save
          </button>
          <button
            onClick={testConnection}
            disabled={test.kind === "testing"}
            className="rounded bg-emerald-700 px-4 py-2 text-sm font-medium hover:bg-emerald-600 disabled:opacity-50"
          >
            {test.kind === "testing" ? "Testing…" : "Test connection"}
          </button>
        </div>

        {test.kind === "ok" && (
          <p className="mt-4 rounded border border-emerald-800 bg-emerald-950 p-3 text-sm text-emerald-300">
            Connected — backend v{test.health.version}, execution mode:{" "}
            <span className="font-semibold">{test.health.execution_mode}</span>
          </p>
        )}
        {test.kind === "error" && (
          <p className="mt-4 rounded border border-red-900 bg-red-950 p-3 text-sm text-red-300">
            Connection failed: {test.message}
          </p>
        )}
      </div>
    </div>
  );
}
