declare global {
  interface Window {
    electronAPI?: { apiPort: number }
  }
}

const port = window.electronAPI?.apiPort ?? 3001

export const apiBase = `http://127.0.0.1:${port}`

export async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${apiBase}${path}`, init)
  if (!res.ok) {
    let message = `${res.status} ${res.statusText}`
    try {
      const body = await res.json() as { error?: string }
      if (body.error) message = body.error
    } catch { /* body isn't JSON, keep the HTTP status */ }
    throw new Error(message)
  }
  return res.json() as Promise<T>
}
