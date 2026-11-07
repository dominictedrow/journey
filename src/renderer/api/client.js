const port = window.electronAPI?.apiPort ?? 3001;
export const apiBase = `http://127.0.0.1:${port}`;
export async function apiFetch(path, init) {
    const res = await fetch(`${apiBase}${path}`, init);
    if (!res.ok) {
        let message = `${res.status} ${res.statusText}`;
        try {
            const body = await res.json();
            if (body.error)
                message = body.error;
        }
        catch { /* body isn't JSON, keep the HTTP status */ }
        throw new Error(message);
    }
    return res.json();
}
