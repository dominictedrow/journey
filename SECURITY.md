# Security model

cwork runs on one person's Mac, does a fair amount of internet access
(Claude web search, Overpass), and is source-available on GitHub. This
document is what that tradeoff looks like in practice.

## Electron hardening

- `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true` on
  the renderer (`app/main.js`).
- A strict Content-Security-Policy is attached to every response
  (`default-src 'self'`, no remote scripts, `object-src 'none'`). The dev
  build relaxes `script-src`/`connect-src` only enough to allow Vite's dev
  server and its Fast Refresh preamble — the production build (loaded from
  a file on disk, no dev server) keeps the strict policy with no
  exceptions.
- All in-app navigation and `window.open` are blocked
  (`will-navigate`/`setWindowOpenHandler`).
- DevTools are disabled outside development (`devTools: isDev`).

## No arbitrary execution from the renderer

The renderer never runs a shell command. IPC (`skills:run`) accepts a
skill **id** only, validated against `app/registry.js` — there's no path by
which renderer-controlled data becomes a shell command or an arbitrary file
path. Python is always spawned with an argument array (`conda run -n
c_work python -m skills.<id> --input <json>`); `shell: true` appears
nowhere in the codebase.

## Secrets never touch the repo

- Any tokens a future skill needs go through `skills/_lib/secrets.py`
  (macOS Keychain via the `security` CLI) — nothing currently in the
  codebase stores a secret in a file, but that helper exists so nothing
  ever has to.
- `.gitignore` excludes `data/`, `.auth/`, and `*.env`.
- There is no API key and no stored password in the default
  configuration — Claude is invoked through the already-authenticated
  `claude -p` CLI, and Overpass needs no key. This is also why an earlier,
  abandoned approach (storing a Google account password to automate portal
  re-login) was rejected outright: it would have been the one exception to
  "nothing here can leak a password."

## Outbound allowlist

`skills/_lib/net.py` only permits requests to a fixed set of hosts
(currently just Overpass, for `places`); anything else raises
`DisallowedHostError` rather than silently connecting. `concepts` and
`releases` do their research through Claude's own web search instead of
this client, so they don't appear on the allowlist at all.

## Claude token/data discipline

- Skills never pass raw HTML to Claude — only pre-filtered, structured
  text (post excerpts, POI names, etc.).
- Every Claude call requests strict JSON against a declared schema.
- Input is truncated above 8KB before being sent.
- Identical prompts are cached for 24h (`skills/_lib/claude.py`) so a
  repeat run doesn't re-spend tokens on unchanged input.
- Anything Claude computes that matters numerically (meal macros) is
  re-validated in Python — Claude's arithmetic is not trusted as the
  source of truth.

## Before every push

```bash
git status --ignored                                            # no session/token files staged
grep -rE "(sk-|ya29\.|AIza|BEGIN .*PRIVATE)" .                   # should be empty over the tracked tree
```
