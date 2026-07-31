# c_work — Personal Automation Suite

## Context

Dom wants a set of Claude-powered skills that automate recurring parts of his life (learning inputs, release watching, cold email, place/meal search), plus a native macOS app that runs them and displays their output — so he never opens a Claude Code terminal to use them. The app also holds a manually-maintained class/assignment tracker (see below) — no scraping, no grades.

Two constraints shape every decision below:

1. **Scripts do the work, Claude does the judgment.** Token spend must be minimal. Deterministic work (fetching, parsing, diffing, filtering, ranking) is Python. Claude is called only where reasoning is genuinely required, and never sees raw HTML.
2. **The codebase stays light.** No dead code, no speculative abstraction, minimum dependencies.

The repo (`/Users/dominictedrow/projects/Code Projects/c_work`) is a fresh git repo on `main` with zero commits. The `c_work` conda env exists but has no Python installed. Node v24.15.0 and Claude Code v2.1.220 are already available.

### Parker Portal automation — investigated, dropped

The original plan automated the Parker Portal (Blackbaud MySchoolApp) via a Playwright-driven authenticated session, pulling assignments/grades as JSON. That spike is done and the approach is **abandoned**:

- Blackbaud issues a short-lived access token at login time that expires within minutes and does not silently refresh — a closed/reopened or headless-reused session just gets `403 INVALID_AUTHORIZATION`, even with valid cookies.
- A fully automated headless re-login was tested and gets partway (filling Blackbaud's login form correctly triggers the real Google SSO redirect) but stalls at Google's account chooser, which reported the account as **signed out** — meaning full automation would require storing Dom's actual Google password and fighting Google's bot-detection on a Workspace-for-Education account. Not worth it, and a different class of secret than anything else in this project.
- Separately confirmed that `claude -p --chrome` **can** open a new tab that inherits Dom's real, already-logged-in Chrome session (same profile, same cookies) — so the mechanism isn't impossible. But building a robust, unattended daily digest around visually reading a live authenticated page was judged more fragile and costly than it's worth right now.

Decision: **no portal scraping, no automated grades/performance tracking.** Replaced with a manual class/assignment tracker built directly into the app UI — see "Classes & Assignments" below. `docs/blackbaud_endpoints.md` and `scripts/auth_portal.py` (the spike artifacts) have been removed. `skills/_lib/browser.py` (generic Playwright persistent-context helper) was kept for `concepts`'s X access at the time, but was later also removed — see the "Personal X login" note under the six skills below, where `concepts` hit the same bot-detection wall.

### Decisions already made (from Q&A)

| Area | Decision |
|---|---|
| App shell | Electron + React + Vite (no new toolchain; Node already present) |
| Claude runtime | `claude -p` headless CLI — $0 marginal cost, no API key to leak |
| Concepts source | Claude web search (X/Twitter, arXiv, Hacker News, GitHub, HuggingFace) — a personal X login was tried and dropped, see below |
| Places / meals | OpenStreetMap Overpass + Claude web search — no API keys, no billing |
| Scheduling | macOS `launchd` — runs whether or not the app is open |
| Meal macros | `skills/meals/config.json` with defaults, editable from Settings |
| Classes / assignments | Manually entered in the UI, stored locally — no scraping, no grades |

---

## Architecture

### The one idea that keeps this light

Every skill emits the **same JSON envelope**. The UI has exactly one renderer that understands it. Adding another skill later means writing Python and adding one line to a registry — no new React code, ever.

```jsonc
{
  "skill": "releases",
  "ok": true,
  "ran_at": "2026-08-01T09:00:00-07:00",
  "summary": "3 releases confirmed this month",
  "sections": [
    { "type": "stat",  "title": "At a glance", "stats": [{"label":"Confirmed","value":3}] },
    { "type": "table", "title": "Upcoming",     "columns": ["Item","Category","Date","Confidence"], "rows": [[...]] },
    { "type": "text",  "title": "Notes",        "body": "markdown..." }
  ],
  "error": null
}
```

Four section types (`stat`, `list`, `table`, `text`) cover every skill. That is the whole UI contract.

The **Classes & Assignments** tracker is not a skill and doesn't go through this envelope — it's plain local CRUD (add/edit/delete), not a "run and get a result" action. See its own section below.

### Repository layout

```
c_work/
├── app/                      # Electron main process (~250 LOC total)
│   ├── main.js               # window, IPC handlers, skill spawner, launchd installer
│   ├── preload.js            # contextBridge — narrow allowlisted surface
│   └── registry.js           # THE skill registry: id, label, kind, schedule, inputs
├── ui/                       # React + Vite renderer
│   ├── index.html
│   ├── main.jsx
│   ├── App.jsx               # sidebar + tab switch (plain state, no router)
│   ├── theme.css             # black/green/blue design tokens
│   ├── components/
│   │   ├── SkillOutput.jsx   # ← the single universal renderer
│   │   ├── RunPanel.jsx      # run button + live stdout log
│   │   └── Card.jsx, Empty.jsx, StatusDot.jsx
│   └── tabs/
│       ├── Home.jsx          # all scheduled-skill results
│       ├── OnDemand.jsx      # generic tab driven by registry
│       ├── Assignments.jsx   # manual class/assignment tracker (CRUD, no skill run)
│       └── Settings.jsx      # auth status, macro config, schedule toggles
├── skills/                   # Python — all real work happens here
│   ├── _lib/
│   │   ├── envelope.py       # build/validate the JSON envelope
│   │   ├── store.py          # sqlite3 (stdlib) — runs, classes, assignments
│   │   ├── claude.py         # subprocess → `claude -p`, JSON out
│   │   ├── net.py            # urllib GET/JSON + outbound domain allowlist
│   │   └── secrets.py        # macOS Keychain via `security` CLI
│   ├── concepts/             # Claude web search → 2 AI + 2 SWE deep dives
│   ├── releases/             # monthly release radar
│   ├── projects/             # AI concept → applied project ideas
│   ├── places/               # Overpass → ranked spots
│   └── meals/                # macro-filtered meal ideas + config.json
├── scripts/
│   └── schedule.py           # install/uninstall launchd agents
├── SECURITY.md
├── README.md
├── environment.yml
├── pyproject.toml
└── package.json
```

`data/`, `.auth/`, and `*.env` are gitignored. Session state and tokens live in `~/Library/Application Support/cwork/`, **outside the repo**, mode `0600`.

### Skill invocation contract

```bash
python -m skills.releases                                       # scheduled skills: no args
python -m skills.places --input '{"query":"calm study spot"}'   # on-demand
```

stdout = the JSON envelope. stderr = human-readable progress lines, which Electron streams into the UI's live log so a run feels instant even when it takes 20 seconds.

---

## Classes & Assignments (manual tracker, not a skill)

Replaces the abandoned Parker Portal scraping. Dom maintains his own class list and assignment due dates directly in the app.

- **No scraping, no grades, no performance tracking** — just classes, assignments, and due dates, entered and edited by hand.
- **Storage:** two tables in `_lib/store.py`'s SQLite database — `classes(id, name, created_at)` and `assignments(id, class_id, title, due_date, done, created_at)`.
- **UI (`ui/tabs/Assignments.jsx`):**
  - Manage classes (add / rename / delete).
  - Add-assignment form: title, class picker, due date.
  - List grouped into Overdue / Due Today / Due This Week / Later / Done, sorted by due date, with inline mark-done and delete.
- **Zero Claude calls, zero external network calls.** Straight Electron main ↔ SQLite CRUD over the same narrow, id-validated IPC surface as everything else — no shell access, nothing skill-registry-shaped about it.

---

## The five skills

### Scheduled (results land on Home)

**1. `concepts`** — daily 7:00 AM. One `claude -p` call with web search (covering X/Twitter, arXiv, Hacker News, GitHub, and HuggingFace) picks **2 AI + 2 software-engineering concepts** worth a deep-dive and writes a brief for each: what it is, why it matters, the one paper or repo to read, and a 30-minute exercise. Avoids repeating a concept covered on a recent day by naming the last ~30 delivered concepts in the prompt and asking Claude not to repeat them.

> **Personal X login — investigated, dropped.** The original plan read Dom's own X timeline (+ any lists) via an authenticated Playwright session, degrading to the free sources above only as a fallback. Tried it in Phase 2/4: `scripts/auth_x.py` opened a real headed browser against a persistent profile, but X's bot detection flagged that automated profile as "not secure" right at the login screen — the same wall the Parker Portal spike hit with Blackbaud's Google SSO. Decision: drop personal-feed reading entirely rather than fight bot detection; Claude's own web search covers the same sources (plus X) without needing Dom's session at all, and runs reliably unattended at 7am regardless of whether Dom is logged into anything. `scripts/auth_x.py` and `skills/_lib/browser.py` (and the `playwright-python` dependency, since nothing else used it) were removed as dead code.

**2. `releases`** — monthly, 1st at 9:00 AM. Checks upcoming releases in all eight of Dom's categories: Pokémon set, One Piece TCG set, MTG set, major-artist album, hyped sneaker, hyped tech, revolutionary AI model/concept, hyped movie. Runs mostly inside one `claude -p` call with web search, returning a strict JSON schema (`item, category, date, confidence, source_url`). Results cached so repeat months only re-check what changed.

### On-demand (own tab, run button)

> **Cold email — dropped.** The original plan had a fourth on-demand skill, `cold_email`: an email template + contacts CSV rendered with `string.Template`, Claude personalizing one opening line per contact, written out as Gmail **drafts only** (never sent automatically). Built and working in Phase 2, but dropped in Phase 4 — before the Gmail OAuth client was ever set up — per Dom's call. `skills/cold_email/`, `scripts/auth_gmail.py`, and the `gmail` optional dependency extra were removed as a result.

**3. `projects`** — Dom names AI concepts or models to grow into. Claude returns 3–5 concrete buildable projects per concept, each with a scope, the specific skill it exercises, a starter repo/dataset, and a difficulty rating.

**4. `places`** — Dom types a natural description ("calm study spot"). Claude converts the description into Overpass QL tags (e.g. `amenity=cafe` + `internet_access=yes` + quiet-leaning tags); the script queries Overpass, filters by distance from Dom's saved home coordinates, and returns a ranked list with map links. Defaults to "near me" unless an area is named.

**5. `meals`** — Dom optionally states a craving ("chicken"). Script loads hard constraints from `skills/meals/config.json` (protein floor, calorie band, required vitamins, avoid-list); Claude proposes meals that satisfy them, with macros and a short recipe. Constraints are enforced in the prompt *and* re-validated in Python — Claude's arithmetic is not trusted.

---

## Phase 5 — additional skills

Six candidates surfaced from a review of the shipped app, chosen because each fit the existing architecture with zero new API keys and no scraping/auth. Dom picked three to start; the other three were deferred rather than built speculatively, then picked up and built in a second pass once the first three had shipped.

**Built, in order: `assignments-prep` → `weather` → `wishlist` → `trips` → `gifts` → `reads`.**

| Skill | Kind | Data source | Claude calls | Notes |
|---|---|---|---|---|
| `assignments_prep` | on-demand | local SQLite (existing `classes`/`assignments` tables) | 1 | Gives the one zero-Claude corner of the app (Assignments) a judgment layer: reads the nearest undone assignment (earliest `due_date`, overdue included) via `_lib/store.py`, asks Claude for 3–6 concrete steps plus a "start here" action. No new network host, no new IPC — reuses the same SQLite file `db.js`'s CRUD writes to. |
| `weather` | scheduled, daily (6:30 AM, ahead of `concepts` at 7:00) | `api.weather.gov` (NOAA — free, keyless, US government) | 1 | Two NOAA calls (resolve gridpoint from home lat/lon, then fetch the period forecast) need zero Claude calls; a "plan your day around this" framing pass adds one, and degrades gracefully (forecast still renders) if that call fails. Reuses the home coordinate already in `config.json` from the `places`/Settings work — no new Settings field. Added `api.weather.gov` to `_lib/net.py`'s allowlist, the first addition since `places` added `overpass-api.de`. |
| `wishlist` | on-demand | Claude web search | 1 | Sibling of `releases` but for one item Dom names instead of the fixed eight categories ("is the PS5 Pro restocked," "did that jacket go on sale"). Built as the pure one-shot query variant discussed below — no new storage, same shape as `places`/`projects`. |
| `trips` | on-demand | Overpass (same host as `places`) | 2 | Same shape as `places` (Overpass + wider radius, "weekend getaway" framing) — shared logic pulled out into `skills/_lib/osm.py` so `places` and `trips` both call the same Overpass-query/rank/format helpers instead of duplicating them. |
| `gifts` | scheduled, weekly (Sunday 8:00 AM) | local SQLite (new `people`/`occasions` tables) | 1 | The `people`/`occasions` tables landed after all, plus a small list-maintenance UI (the "People" sidebar tab, `ui/tabs/Gifts.jsx` — plain CRUD, zero Claude calls, same shape as Assignments). This skill is the scheduled judgment layer on top: finds occasions landing within 21 days, asks Claude for gift ideas for all of them in one batched call. |
| `reads` | scheduled, monthly (1st, 10:00 AM, after `releases` at 9:00) | Claude web search | 1 | Book-world sibling of `concepts` — same "list recently-delivered items in the prompt to avoid repeats" pattern, one call per run. |

### Decisions made

- **`weather` is a daily scheduled card on Home**, not on-demand-only — same pattern as `concepts`/`releases`, scheduled 30 minutes ahead of `concepts` so both are ready before Dom's day starts.
- **`wishlist` is a one-shot on-demand query**, not a persisted list — Dom types an item, gets a status check, nothing is stored.
- **`gifts` did end up needing the "Lists pattern" question** the first pass punted on — it's the second list-backed skill after Assignments, with its own tab (`ui/tabs/Gifts.jsx`) rather than a generalized Lists abstraction. No third consumer has shown up yet, so that generalization still hasn't been worth building.

### Incremental cost per skill (actual)

Matched the Phase 2 shape: each skill is a `skills/<id>/__main__.py` plus one `app/registry.js` entry; `trips`/`gifts` also needed one new `_lib` module (`osm.py`) and two new SQLite tables (`people`/`occasions`) respectively, but no new IPC surface beyond that pattern and no new UI code besides `gifts`'s own People tab. `weather` needed a `_lib/net.py` allowlist change; `trips` reused `places`'s existing entry. Any scheduled skill (`concepts`/`releases`/`weather`/`gifts`/`reads`) needs `scripts/schedule.py install` (re-)run to pick up its launchd agent — not run automatically as part of a build, since it's a persistent system-level change outside the repo.

---

## Claude token discipline

| Skill | Claude calls per run | What it sees |
|---|---|---|
| `concepts` | 1 | the prompt + recently-delivered concept names |
| `releases` | 1 | search results only |
| `places` | 2 | the query; then ~30 POI names |
| `meals`, `projects` | 1 | the prompt + config |
| Classes & Assignments | **0** | — (plain CRUD) |

Rules enforced in `_lib/claude.py`: never pass raw HTML; always request `--output-format json` against a declared schema; truncate any input over 8KB; cache identical prompts for 24h in SQLite.

---

## Dependencies (deliberately minimal)

### Conda — **just Python itself**

```yaml
# environment.yml
name: c_work
channels: [conda-forge]
dependencies:
  - python=3.11
```

Everything else uses the Python standard library:

| Need | Stdlib module used |
|---|---|
| HTTP / JSON (Overpass) | `urllib.request` + `json` |
| Storage | `sqlite3` |
| Keychain, Claude CLI, notifications | `subprocess` |

`playwright-python` was in this list from Phase 0 through most of Phase 2, as the authenticated HTTP client for `concepts`'s planned X-session reading. Removed once that approach was tried and dropped (see the "Personal X login" note above) — nothing else in the codebase used it. `csv`/`string.Template` and the optional `gmail` extra (`google-api-python-client`, `google-auth-oauthlib`) were similarly dropped along with `cold_email` in Phase 4 (see the "Cold email — dropped" note above) — nothing else used them either.

### Node — 2 runtime, 4 dev

`react`, `react-dom` · `electron`, `electron-builder`, `vite`, `@vitejs/plugin-react`.

No router (tabs are component state), no state manager, no UI kit, no Tailwind. The theme is ~120 lines of CSS custom properties.

---

## Design

Black background, blue and green accents, output-first.

```css
--bg:        #07090B;   --surface:  #0E1216;   --border: #1B2228;
--text:      #E6EDF3;   --muted:    #7D8894;
--green:     #22E88A;   /* success, positive deltas, primary action */
--blue:      #3B9EFF;   /* links, secondary accents, selection    */
--red:       #FF5C5C;   /* overdue assignments, failures only     */
```

- Frameless window with `titleBarStyle: 'hiddenInset'` so it reads as a real Mac app.
- Left sidebar: Home, then one row per skill with a status dot (green = fresh, blue = running, muted = stale, red = failed) and last-run time, plus a dedicated Assignments row (no status dot — it's not a run).
- **Home** is a grid of cards, one per scheduled skill, rendered from cached SQLite — it never waits on the network, so it paints instantly on launch.
- Pressing Run gives feedback in the same frame: the button switches to a live streaming log from stderr, then the result renders in place.
- System font stack (`-apple-system`), generous spacing, no shadows, 1px borders.

---

## Security

Given this goes on GitHub and does a lot of authenticated internet access:

- **Electron hardening** — `contextIsolation: true`, `nodeIntegration: false`, `sandbox: true`, strict CSP, all in-app navigation and `window.open` blocked, DevTools off in production.
- **No arbitrary execution from the renderer.** IPC accepts a skill *id* only, validated against `app/registry.js`. Python is spawned with an argument array; `shell=True` appears nowhere.
- **Secrets never touch the repo.** Anything a future skill stores via `skills/_lib/secrets.py` lives in the macOS Keychain, not a file. Nothing is committed; `.gitignore` covers `data/`, `.auth/`, and `*.env`.
- **No API keys, no stored passwords** in the default configuration — the CLI-based Claude runtime and keyless Overpass choice mean there is nothing to leak. (This is also why the "store Dom's Google password for automated portal re-login" option was rejected — it would have been the one exception.)
- **Outbound allowlist** in `_lib/net.py`; unexpected hosts raise rather than silently connect.
- `SECURITY.md` documents the model; README setup uses placeholder data so the public repo contains none of Dom's personal information.

---

## Build order

Each phase ends somewhere runnable.

**Phase 0 — Foundation**
`environment.yml`, `pyproject.toml`, `package.json`, `.gitignore`, `_lib/` (envelope, store, claude, net, secrets). Ship a trivial `skills/hello` proving the envelope round-trips end to end.

**Phase 1 — Electron shell + UI, including Classes & Assignments**
Main process, preload, registry, theme, `SkillOutput.jsx`, Home, the generic on-demand tab, and the Assignments tab (classes CRUD, assignment CRUD, due-date grouping). This phase needs no external auth at all and delivers a genuinely useful app on its own.

**Phase 2 — Remaining skills**
`concepts`, `releases`, `places`, `meals`, `projects`. Each is Python plus one registry line — no additional UI work. (`concepts` originally planned a personal-X-login step here, and a sixth skill `cold_email` also shipped in this phase; both were tried/built and then dropped in Phase 4 — see the notes under "The five skills.")

**Phase 3 — Scheduling, packaging, docs**
`scripts/schedule.py` writes the launchd plists (for `concepts`/`releases`; Assignments needs no schedule, it's manual); `electron-builder` produces a signed-if-possible `.app` and DMG; README, SECURITY.md, screenshots.

**Phase 4 — Settings tab**
`ui/tabs/Settings.jsx` went from an "Empty" stub (deferred at Phase 1) to actually editing the two configs that needed hand-editing: home coordinate and meal macros. Both route through validated IPC (`app/settings.js`) into `~/Library/Application Support/cwork/config.json`, never into a tracked or packaged file — `skills/meals/config.json` stays a read-only shipped default template, exactly like `places` already treated the home coordinate.

**Phase 5 — additional skills**
`assignments_prep`, `weather`, `wishlist` — see "Phase 5 — additional skills" above. Each is Python plus one registry line, same size as a Phase 2 skill; `weather` also added one `_lib/net.py` allowlist entry.

---

## What I need from you (setup-time, not blocking the build)

Nothing here blocks starting — these come up at specific phases and each takes about a minute.

| When | What I need |
|---|---|
| Phase 1 | Your home address or neighborhood, for the `places` "near me" default. |
| Phase 2 | Any changes to the default meal macros — the shipped defaults are protein ≥ 40 g, 500–900 cal, vitamins D/B12/Iron, no allergens. Also tell me any allergies. |

---

## Verification

- **Per skill:** `python -m skills.<name>` prints a valid envelope. A tiny `_lib/envelope.py` validator rejects malformed output, so a broken skill fails loudly rather than rendering blank.
- **UI:** launch with `npm run dev`, confirm Home paints from cache in under 300 ms with the network off; confirm each on-demand tab streams a live log and renders its result; confirm the Assignments tab can add/edit/delete a class and an assignment and persists across an app restart.
- **Scheduling:** `launchctl list | grep cwork`, plus a forced run via `launchctl kickstart` to confirm the agents fire outside the app.
- **Security:** `git status --ignored` shows no session/token/CSV files staged; confirm `grep -rE "(sk-|ya29\.|AIza|BEGIN .*PRIVATE)" .` over the tracked tree is empty before the first push.
- **End to end:** close the app, wait for a scheduled run, reopen — the new results should already be on Home.
