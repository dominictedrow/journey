"""Template x contacts CSV -> Gmail drafts (draft-only, never sent).

Invocation:
    python -m skills.cold_email --input '{"template_path":"/path/to/template.txt",
                                           "contacts_csv_path":"/path/to/contacts.csv",
                                           "mode":"preview"}'

`template_path` is a path to a plain-text file containing {{variable}} placeholders
(double-brace syntax). `contacts_csv_path` is a path to a CSV with exactly the
columns `name, email`. `mode` is "preview" (default, zero Gmail dependency) or
"draft" (creates Gmail drafts only -- never sends).

Per plan.md: template x contact rows are rendered with `string.Template`
(stdlib, no new dependency), subclassed to recognize {{variable}} instead of
the stdlib default $variable / ${variable} syntax. See BraceTemplate below.
"""

import argparse
import csv
import json
import re
import string
from pathlib import Path

from .._lib import claude, envelope

_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"


class BraceTemplate(string.Template):
    """string.Template subclass recognizing {{variable}} instead of $variable.

    string.Template requires `pattern` to be a regex exposing four named
    groups: escaped, named, braced, invalid (see stdlib docs on subclassing).
    We only ever use the braced form here; `named` is wired to an alternative
    that can never match ((?!) is a negative lookahead on the empty string,
    which always succeeds, so its negation always fails) purely to satisfy
    that contract without introducing a second (redundant) placeholder syntax.
    """

    delimiter = "{{"  # not used to build `pattern` (we override that directly), but
    # `Template.safe_substitute`'s escaped-match branch returns `self.delimiter` literally,
    # so this must match our actual placeholder opener for {{{{ ... }}}} escaping to work.
    pattern = rf"""
    \{{\{{(?:
      (?P<escaped>\{{\{{)                  |
      (?P<named>(?!))                      |
      (?P<braced>{_IDENTIFIER})\}}\}}       |
      (?P<invalid>)
    )
    """


def render_template(template_text, row):
    """Render one contact row into the template. Missing vars are left as
    literal {{var}} (safe_substitute), so a bad row surfaces visibly instead
    of crashing the whole batch."""
    return BraceTemplate(template_text).safe_substitute(**row)


def load_template(template_path):
    path = Path(template_path)
    if not path.is_file():
        raise FileNotFoundError(f"template file not found: {template_path}")
    return path.read_text(encoding="utf-8")


def load_contacts(contacts_csv_path):
    path = Path(contacts_csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"contacts CSV not found: {contacts_csv_path}")

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        if "name" not in fieldnames or "email" not in fieldnames:
            raise ValueError(
                f"contacts CSV must have columns 'name, email', got: {reader.fieldnames}"
            )
        rows = [
            {"name": (r.get("name") or "").strip(), "email": (r.get("email") or "").strip()}
            for r in reader
        ]

    if not rows:
        raise ValueError("contacts CSV has no data rows")

    return rows


def _parse_claude_json(raw_result):
    """Claude's `result` text should be strict JSON; strip stray markdown fences defensively."""
    text = raw_result.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def build_personalization_prompt(contacts):
    """One batched prompt across ALL contacts -- never one Claude call per
    contact (token discipline, per plan.md)."""
    payload = json.dumps(
        [
            {"index": i, "name": c["name"], "email": c["email"], "body_preview": c["body"][:400]}
            for i, c in enumerate(contacts)
        ],
        ensure_ascii=False,
    )
    return f"""Dom is sending a batch of cold emails. Here is a JSON list of
contacts, each with their name, email, and a preview of the rendered email
body they'll receive: {payload}

For EACH contact, write one short, personalized opening line (one sentence,
warm but professional, no generic filler like "I hope this finds you well")
that could be inserted at the top of their email body.

Respond with ONLY strict JSON (no markdown fences, no commentary) matching
exactly this schema:
{{
  "openers": [
    {{"index": number, "opener": "string"}}
  ]
}}"""


def personalize(contacts):
    """Batched Claude call to get one opening line per contact. Returns a dict
    index -> opener string. On any failure, returns {} so callers can proceed
    without personalization rather than failing the whole run."""
    if not contacts:
        return {}

    prompt = build_personalization_prompt(contacts)
    envelope.progress(f"cold_email: asking Claude for {len(contacts)} personalized opener(s)")

    try:
        response = claude.ask(prompt)
        parsed = _parse_claude_json(response["result"])
        openers = {o["index"]: o["opener"] for o in parsed.get("openers", [])}
    except (KeyError, json.JSONDecodeError, TypeError, ValueError) as e:
        envelope.progress(f"cold_email: personalization failed, proceeding without it: {e}")
        return {}

    return openers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="{}")
    args = parser.parse_args()

    payload = json.loads(args.input)
    template_path = payload.get("template_path") or ""
    contacts_csv_path = payload.get("contacts_csv_path") or ""
    mode = (payload.get("mode") or "preview").strip().lower() or "preview"

    if mode not in ("preview", "draft"):
        envelope.emit(
            "cold_email",
            ok=False,
            summary=f"invalid mode {mode!r}, must be 'preview' or 'draft'",
            sections=[],
            error=f"invalid mode: {mode!r}",
        )
        return

    envelope.progress(f"cold_email: mode={mode}")

    try:
        template_text = load_template(template_path)
        contacts = load_contacts(contacts_csv_path)
    except (FileNotFoundError, ValueError) as e:
        envelope.progress(f"cold_email: input error: {e}")
        envelope.emit(
            "cold_email",
            ok=False,
            summary="failed to load template or contacts",
            sections=[],
            error=str(e),
        )
        return

    envelope.progress(f"cold_email: rendering template for {len(contacts)} contact(s)")

    for c in contacts:
        c["body"] = render_template(template_text, c)

    openers = personalize(contacts)
    for i, c in enumerate(contacts):
        opener = openers.get(i)
        c["final_body"] = f"{opener}\n\n{c['body']}" if opener else c["body"]

    if mode == "preview":
        sections = [
            {
                "type": "table",
                "title": "Preview -- would send to these contacts",
                "columns": ["Name", "Email", "Body preview"],
                "rows": [
                    [c["name"], c["email"], c["final_body"][:200].replace("\n", " ")]
                    for c in contacts
                ],
            }
        ]
        envelope.progress("cold_email: preview complete")
        envelope.emit(
            "cold_email",
            ok=True,
            summary=f"Previewed {len(contacts)} email(s), nothing sent or drafted",
            sections=sections,
        )
        return

    # mode == "draft" -- only path that touches Gmail, and only creates drafts.
    envelope.progress("cold_email: draft mode -- creating Gmail drafts")

    from .._lib import secrets  # noqa: local import, draft-mode only

    token_json = secrets.get("gmail_token")
    if not token_json:
        envelope.emit(
            "cold_email",
            ok=False,
            summary="no Gmail OAuth token found -- run scripts/auth_gmail.py once first",
            sections=[],
            error="missing gmail_token in keychain",
        )
        return

    try:
        import base64
        from email.mime.text import MIMEText

        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
    except ImportError as e:
        envelope.emit(
            "cold_email",
            ok=False,
            summary="Gmail draft mode requires the optional 'gmail' extra",
            sections=[],
            error=f"missing dependency: {e}",
        )
        return

    creds = Credentials.from_authorized_user_info(json.loads(token_json))
    gmail = build("gmail", "v1", credentials=creds)

    created = []
    failed = []
    for c in contacts:
        message = MIMEText(c["final_body"])
        message["to"] = c["email"]
        message["subject"] = f"Following up, {c['name']}"
        raw = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")

        try:
            gmail.users().drafts().create(
                userId="me", body={"message": {"raw": raw}}
            ).execute()
            created.append(c)
        except Exception as e:  # noqa: broad -- surface per-contact failures, keep batch going
            envelope.progress(f"cold_email: failed to draft for {c['email']}: {e}")
            failed.append((c, str(e)))

    sections = [
        {
            "type": "table",
            "title": f"Gmail drafts created ({len(created)})",
            "columns": ["Name", "Email"],
            "rows": [[c["name"], c["email"]] for c in created],
        }
    ]
    if failed:
        sections.append(
            {
                "type": "text",
                "title": f"Failed to draft ({len(failed)})",
                "body": "\n".join(f"- {c['email']}: {reason}" for c, reason in failed),
            }
        )

    envelope.progress(f"cold_email: created {len(created)} draft(s), {len(failed)} failed")
    envelope.emit(
        "cold_email",
        ok=len(failed) == 0,
        summary=f"Created {len(created)} Gmail draft(s), {len(failed)} failed",
        sections=sections,
        error=None if not failed else f"{len(failed)} draft(s) failed to create",
    )


if __name__ == "__main__":
    main()
