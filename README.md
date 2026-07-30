# cwork

A personal automation suite: a set of Claude-powered skills that handle
recurring parts of daily life (learning digests, release watching, cold
email, place/meal search), plus a native-feeling macOS app that runs them
and displays their output. You never open a terminal to use it day to day.

Every skill is a small Python module that prints one JSON envelope; the app
has exactly one renderer that understands that envelope. Adding a skill
later means writing Python and adding one entry to a registry — no new UI
code.

## Skills

| Skill | Kind | What it does |
|---|---|---|
| `concepts` | scheduled, daily | Reads your X timeline (+ configured lists), picks 2 AI + 2 software-engineering concepts worth a deep-dive, writes a brief for each. Falls back to arXiv/Hacker News/GitHub/HuggingFace if X isn't reachable. |
| `releases` | scheduled, monthly | Checks upcoming releases across categories you care about (an album, a game set, a movie, a gadget — configure your own list) via Claude web search. |
| `cold_email` | on-demand | Template + contacts CSV → personalized Gmail **drafts only**. Nothing is ever sent automatically. |
| `projects` | on-demand | Turns a named concept/model into 3-5 concrete practice projects. |
| `places` | on-demand | Natural-language query ("calm study spot") → ranked nearby places via OpenStreetMap. |
| `meals` | on-demand | Macro-constrained meal ideas + recipes, constraints enforced in Python, not just prompted. |

Plus a manual **Classes & Assignments** tracker (add/edit/delete, due-date
grouping) — plain local CRUD, no scraping, no external calls.

## Setup

Requirements: [conda](https://docs.conda.io), Node 20+, and the
[Claude Code CLI](https://claude.com/claude-code) already authenticated
(`claude` on your PATH).

```bash
conda env create -f environment.yml
conda run -n c_work playwright install chromium

npm install
npm run dev          # UI dev server
npm run electron     # in another terminal, launches the app shell
```

### Per-skill setup

- **`places`** — put your home coordinates in
  `~/Library/Application Support/cwork/config.json`:
  ```json
  { "home": { "lat": 0.0, "lon": 0.0, "label": "Your area" } }
  ```
- **`meals`** — edit `skills/meals/config.json` (protein floor, calorie
  band, required vitamins, avoid-list) to match your own targets/allergies.
- **`concepts`** — run `python scripts/auth_x.py` once to log into X in a
  real (headed) browser window; the session is saved to
  `~/Library/Application Support/cwork/x_profile`. Optionally list X list
  URLs to also read from under `"concepts": {"x_lists": [...]}` in the same
  `config.json` as above. If the saved session isn't valid at run time,
  `concepts` automatically falls back to free sources instead of failing.
- **`cold_email`** (draft mode only — `--dry-run`/preview needs nothing) —
  create a Desktop-app OAuth client with the Gmail API enabled at
  https://console.cloud.google.com/apis/credentials, save its downloaded
  JSON as `.auth/gmail_client_secret.json`, then run
  `python scripts/auth_gmail.py` once. Requires the optional extra:
  `pip install -e ".[gmail]"`.

## Running a skill directly

```bash
python -m skills.releases                                      # scheduled skills: no args
python -m skills.places --input '{"query":"calm study spot"}'   # on-demand skills: JSON input
```

stdout is the JSON envelope; stderr is a human-readable progress log (the
app streams this live while a skill runs).

## Scheduling

Scheduled skills (`concepts`, `releases`) run via macOS `launchd`, whether
or not the app is open:

```bash
python scripts/schedule.py install     # write + load launchd agents
python scripts/schedule.py status      # launchctl list | grep com.cwork
python scripts/schedule.py uninstall   # unload + remove them
```

## Packaging

```bash
npm run dist   # vite build + electron-builder → release/mac-arm64/cwork.app + .dmg
```

Signs with a local Developer ID certificate if one is present in your
Keychain, otherwise produces an ad-hoc-signed build (Apple Silicon requires
at least that to run at all). This isn't set up for public distribution —
it's a way to get a double-clickable app on your own Mac.

## Design

Black background, blue and green accents, output-first — see `plan.md` for
the full rationale and architecture notes.
