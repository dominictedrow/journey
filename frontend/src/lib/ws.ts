import { useEffect, useRef, useState } from "react";

import type { LiveFrame } from "@/lib/api";
import { useSettings } from "@/stores/settings";

/**
 * WebSocket client for live portfolio/risk updates. The backend `/ws/live` route pushes a
 * full state frame on connect and every interval; this derives the ws:// URL from the
 * configured backend URL.
 */
export function createBackendSocket(path: string): WebSocket {
  const base = useSettings.getState().backendUrl.replace(/^http/, "ws");
  return new WebSocket(`${base}${path}`);
}

export interface LiveState {
  frame: LiveFrame | null;
  connected: boolean;
}

/**
 * Subscribe to `/ws/live` and return the latest frame. Reconnects automatically after a drop
 * (e.g. the backend restarting), and follows changes to the configured backend URL.
 */
export function useLiveState(mode = "paper"): LiveState {
  const backendUrl = useSettings((s) => s.backendUrl);
  const [frame, setFrame] = useState<LiveFrame | null>(null);
  const [connected, setConnected] = useState(false);
  const retry = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    let closed = false;
    let socket: WebSocket | null = null;

    const connect = () => {
      if (closed) return;
      socket = createBackendSocket(`/ws/live?mode=${mode}`);
      socket.onopen = () => setConnected(true);
      socket.onmessage = (event) => {
        try {
          setFrame(JSON.parse(event.data) as LiveFrame);
        } catch {
          // ignore malformed frames
        }
      };
      socket.onclose = () => {
        setConnected(false);
        if (!closed) retry.current = setTimeout(connect, 2000);
      };
      socket.onerror = () => socket?.close();
    };
    connect();

    return () => {
      closed = true;
      if (retry.current) clearTimeout(retry.current);
      socket?.close();
    };
  }, [backendUrl, mode]);

  return { frame, connected };
}
