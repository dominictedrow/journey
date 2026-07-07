import { create } from "zustand";
import { persist } from "zustand/middleware";

interface SettingsState {
  backendUrl: string;
  setBackendUrl: (url: string) => void;
}

// The backend URL is the only client-side setting: the same app (or .dmg) points at the
// Windows test box or the Beelink just by changing it here.
export const useSettings = create<SettingsState>()(
  persist(
    (set) => ({
      backendUrl: "http://localhost:8000",
      setBackendUrl: (backendUrl) => set({ backendUrl: backendUrl.replace(/\/+$/, "") }),
    }),
    { name: "copytrader-settings" },
  ),
);
