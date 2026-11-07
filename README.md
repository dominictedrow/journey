# cold_email: Gmail mail merge

> **Part of [journey](https://github.com/dominictedrow/journey)**, Dom's ML and software portfolio.
>
> **How it was built: 🤖 vibe coded.** AI coding agents (mostly Claude Code) wrote the code. I picked the goal, made the design calls, and steered and tested the work.
>
> **When:** 2026-06 · **Branch:** `vibe/cold_email` · [All vibe projects](https://github.com/dominictedrow/journey/tree/vibe-projects)

A small, dependency-free Python script that sends personalized sponsorship emails from Gmail. It merges a CSV with a template and paces, caps and logs every send so it can resume without double-sending. Built for Parker Pickleball Association outreach.

Usage: `README.md` below.

---

# Cold Email — Gmail Mail Merge

A small, dependency-free Python script that sends personalized sponsorship
emails from your Gmail. Feed it a CSV of recipients and a template with
`{{name}}` / `{{organization}}` placeholders; it merges, paces, caps, and logs
each send so your outreach lands in inboxes instead of spam — and so you can
stop and re-run any time without double-sending.

**No database, no `pip` installs.** Python standard library only. The only state
on disk is `sent_log.csv`, a plain CSV used to resume.

---

## 1. One-time setup: Gmail App Password

A free Gmail caps at ~500 sends/day and requires an *App Password* (not your
normal password) for scripts.

1. Turn on **2-Step Verification**: Google Account → Security → 2-Step
   Verification.
2. Create an **App Password**: Google Account → Security → 2-Step Verification →
   **App passwords** → generate one. You'll get a 16-character code.
3. Keep it secret. You can revoke it from the same page at any time.

## 2. Provide your credentials (every new terminal session)

Secrets are read from the environment — never stored in any file or committed:

```bash
export GMAIL_ADDRESS="you@gmail.com"
export GMAIL_APP_PASSWORD="abcd efgh ijkl mnop"   # the 16-char App Password
```

## 3. Fill in your content

- **`config.json`** — set `from_name` to your club's name. Tune
  `min_delay_seconds` / `max_delay_seconds` (pacing between sends) and
  `daily_cap` (default 400, safely under Gmail's ~500) if you like.
- **`template.txt`** — the **first non-blank line is the subject**, everything
  after it is the body. Use `{{column}}` anywhere to drop in a value from the
  CSV.
- **`recipients.csv`** — first row is the header. The `email` column is always
  required. Beyond that, add **any columns you want** (`name`, `organization`,
  `city`, `owner_first_name`, ...) and reference each one as `{{columnname}}` in
  the template. The template decides which columns are mandatory: every
  placeholder it uses must exist as a column and be filled in for each recipient.
  Extra columns the template doesn't use are ignored.

Placeholders are matched by exact column name (surrounding spaces like
`{{ city }}` are fine). If the template references a column the CSV doesn't have,
the script stops with a clear error before sending anything; a row with a blank
value for a required placeholder is skipped and reported.

## 4. Send — work up gradually

```bash
# 1. Preview the merge. Sends nothing, needs no password.
python3 send_emails.py --dry-run

# 2. Send one real email to yourself and confirm it lands in the INBOX, not spam.
python3 send_emails.py --test

# 3. Send a small real batch to confirm pacing + logging look right.
python3 send_emails.py --limit 5

# 4. Full run. Stop with Ctrl-C any time; re-run to resume where you left off.
python3 send_emails.py
```

`python3 send_emails.py --help` lists every flag.

---

## How resuming works

Every attempt is appended to `sent_log.csv` (`email,timestamp,status`) and
flushed to disk immediately. On the next run the script loads that log into
memory and **skips any address already marked `sent`** — so a crash, a Ctrl-C, a
hit daily cap, or just running again is always safe.

Failed addresses are retried, but not forever:

- An address Gmail refuses outright (bad recipient/sender) is logged
  `failed_permanent` and **never retried** — repeatedly mailing a dead address
  wastes your daily cap and looks like spam.
- A transient failure (timeout, temporary error) is retried on later runs, up to
  3 attempts, then given up on.

Obviously-malformed emails in the CSV (no `@`, spaces, no domain dot) are skipped
before any send is attempted and reported as invalid rows.

To start completely fresh, delete `sent_log.csv`.

## Flags

| Flag | Effect |
|------|--------|
| `--dry-run` | Render merged emails to the terminal; send nothing (no password needed). |
| `--test` | Send one merged email to your own `GMAIL_ADDRESS`, then exit. |
| `--limit N` | Send (or render) at most N emails this run, on top of the daily cap. |
| `--config PATH` | Use a different config file (default `config.json`). |
| `--recipients PATH` | Override the recipients CSV path. |
| `--template PATH` | Override the template path. |

---

## Legal: CAN-SPAM

Cold commercial email in the US must follow the CAN-SPAM Act. Put these in your
`template.txt` body:

- A **clear opt-out** line (e.g. "Reply with UNSUBSCRIBE and we'll remove you").
- A **real physical mailing address** for your club.
- An accurate, non-deceptive subject line and "From" name.

Honor opt-out requests promptly (remove them from `recipients.csv`).
