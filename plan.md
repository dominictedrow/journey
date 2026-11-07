# Design: Gmail Cold-Email Mail Merge (`cold_email`)

## Context

Dom runs a club that raises money via tournaments and needs to reach a lot of
local businesses for sponsorship. Today's options are pasting names into Gmail by
hand (slow, error-prone) or a paid third-party tool. He wants a small Python
script he controls: feed it a CSV of recipients + a template with placeholders,
and it sends personalized emails from his Gmail.

**The risk that shapes the design:** "email a lot of people very fast" is exactly
the fingerprint Gmail's spam filter flags, and a free Gmail caps at ~500/day. If
emails land in spam, the outreach raises $0. So the script sends *personalized and
paced*, and the code itself is *tight* — login once, reuse one connection, stream
the CSV, O(1) resume lookups. "Fast" = efficient code + no wasted work, not
hammering Gmail.

**Locked decisions (office-hours):** Python standard library only (`smtplib`,
`email`, `csv`, `argparse`, `string`, `random`, `time`) + a Gmail **App Password**.
Zero `pip` installs. Features: send-to-self `--test`, rate limiting + daily cap,
resumable send log, plus `--dry-run` preview.

## Final project layout

```
cold_email/
  send_emails.py        # all logic
  config.json           # non-secret settings (from name, delays, daily cap, paths)
  template.txt          # Subject: line + body with {{name}} / {{organization}}
  recipients.csv        # Dom provides: email,name,organization
  sent_log.csv          # auto-created: email,timestamp,status  (resume state)
  README.md             # App Password setup + usage
```

Secrets stay out of files: `GMAIL_ADDRESS` and `GMAIL_APP_PASSWORD` come from
environment variables — nothing sensitive written to disk or committed.

## Key format decisions (apply across all phases)

- **Template placeholders use `{{name}}` / `{{organization}}`**, filled with plain
  `str.replace`. Chosen over `string.Template`/`str.format` because fundraising
  copy contains literal `$` and `{}` that would break those; `{{...}}` never
  collides.
- **`template.txt`**: first non-blank line = subject, remainder = body.
- **`recipients.csv`**: header `email,name,organization`, read with
  `csv.DictReader` (streamed, not loaded whole).
- **Plain-text email** via `email.message.EmailMessage` — looks personal and is
  the best deliverability choice for cold outreach.

---

## Phase 1 — Core merge + send engine

Goal: a recipient CSV + template turns into correctly merged, personalized emails
that actually land in an inbox. End of phase = you can send a real personalized
batch (small, manual count).

Build:
1. **Config + secrets loader.** Read `config.json`; read `GMAIL_ADDRESS` /
   `GMAIL_APP_PASSWORD` from env. Fail with a clear one-line message if the App
   Password is unset.
2. **Template parser.** Split `template.txt` into subject + body once at startup.
3. **CSV reader + per-row validation.** `csv.DictReader` iteration; a row missing
   any of email/name/organization is skipped and reported (never send "Hi
   {{name}}").
4. **Merge function.** `subject`/`body` → filled strings via `{{...}}` replace.
   Pure function, no I/O — trivially unit-testable and reused by `--dry-run`,
   `--test`, and the real loop.
5. **Message builder + sender.** Build `EmailMessage` (From with `from_name`, To,
   Subject, plain-text body); send over `smtplib.SMTP_SSL("smtp.gmail.com", 465)`.
6. **`--dry-run`** — render first N merged emails to terminal, send nothing.
7. **`--test`** — send ONE real merged email to your own `GMAIL_ADDRESS`, exit.

Phase 1 verification:
- `--dry-run` on a 2-row CSV: both render, names/orgs filled, no leftover `{{...}}`.
- `--test`: one email arrives in your inbox, correct subject/body, **not** in spam.

---

## Phase 2 — Safety, resilience, resumability

Goal: the script can be run on the real list, stopped, crash, or hit a cap, and
re-run safely with zero double-sends. This is the deliverability + reliability
layer.

Build:
1. **Resumable send log.** Load `sent_log.csv` once into a `set` (O(1) lookups);
   append `email,timestamp,status` immediately after each attempt and `flush()`
   so a crash never loses state. Already-sent addresses are skipped on re-run.
2. **Rate limiting.** `time.sleep(random.uniform(min_delay, max_delay))` between
   sends; delays from config. Intentional pacing for deliverability — keep it,
   but configurable.
3. **Daily cap.** Count today's `sent` rows from the log at startup; stop with a
   clear "resume tomorrow" message at `daily_cap` (default 400, under Gmail's
   ~500). `--limit N` caps a single run on top of that.
4. **Per-recipient fault isolation.** `try/except smtplib.SMTPException` per send:
   log `failed:<reason>`, continue. One bad address never crashes the run.
5. **Graceful stop.** Trap `KeyboardInterrupt` (Ctrl-C) so the current log line is
   flushed and a summary prints before exit.

Phase 2 verification:
- `--limit 2` on owned inboxes: both arrive, log has two `sent` rows; immediate
  re-run sends **zero** (resume works).
- Malformed row (missing org): skipped + logged, run continues.
- Ctrl-C mid-run: clean summary, log intact, re-run resumes correctly.

---

## Phase 3 — Efficiency, polish, docs

Goal: make it run clean and efficient, and make it usable by a non-engineer.

Build:
1. **Single persistent SMTP connection (main efficiency win).** Connect + login
   **once**, reuse the connection for the whole batch instead of reconnecting per
   email. Gmail can drop an idle connection during long pauses, so wrap sends with
   **auto-reconnect**: on `SMTPServerDisconnected`, reconnect once and retry the
   current message before marking it failed.
2. **Efficient hot path.** Stream the CSV (no full load), keep the sent-set in
   memory, do the cheap `{{...}}` replaces, append-and-flush the log line. No
   re-reading files, no per-row reconnects, minimal allocations.
3. **CLI finalization** via `argparse`: `--dry-run`, `--test`, `--limit N`,
   `--config`, `--recipients`, `--template`. `--help` reads cleanly.
4. **Run summary.** On completion: counts of sent / failed / skipped + where the
   log is.
5. **README.md**: enable 2FA → create App Password → `export` the two env vars →
   fill CSV + template → `--dry-run` → `--test` → `--limit 5` → full run. Include
   a CAN-SPAM note: add a real opt-out line + physical address to the template.
6. **`config.json` shipped with sane defaults:**
   ```json
   {
     "from_name": "Club Name",
     "min_delay_seconds": 30,
     "max_delay_seconds": 90,
     "daily_cap": 400,
     "recipients_path": "recipients.csv",
     "template_path": "template.txt",
     "sent_log_path": "sent_log.csv"
   }
   ```

Phase 3 verification (full end-to-end):
- `python3 send_emails.py --help` lists all flags clearly.
- Full run on a 3-row owned-inbox CSV: all arrive personalized and not in spam,
  one SMTP login in the logs (not three), summary prints correct counts.
- Simulate a dropped connection (long delay) → auto-reconnect recovers, no
  duplicate send.

## Out of scope (separate work if ever wanted)

- Gmail API/OAuth, reply/bounce reading (Approach B — unneeded for a club tool).
- HTML emails, open/click tracking (hurts deliverability and trust here).
- Scheduling/cron, multi-account rotation.
