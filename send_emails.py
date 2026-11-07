#!/usr/bin/env python3
"""Gmail cold-email mail merge.

Reads a CSV of recipients and a template with {{name}} / {{organization}}
placeholders, merges them into personalized plain-text emails, and sends them
from your Gmail over a single persistent SMTP_SSL connection — paced, capped,
and resumable so a free Gmail account stays under its limits and out of spam.

Features:
  --dry-run        render merged emails to the terminal, send nothing
  --test           send ONE merged email to your own GMAIL_ADDRESS, then exit
  --limit N        send at most N emails this run (on top of the daily cap)
  --config PATH    config JSON (default: config.json)
  --recipients PATH / --template PATH   override paths from config

Reliability:
  * Resumable send log (CSV): every attempt is appended + flushed immediately,
    so a crash or Ctrl-C never loses state and a re-run never double-sends.
  * Rate limiting: a random pause between sends (deliverability, not speed).
  * Daily cap: stops before Gmail's ~500/day ceiling and says "resume tomorrow".
  * Per-recipient fault isolation: one bad address is logged and skipped, never
    crashing the run.
  * Single persistent SMTP connection with auto-reconnect on a dropped socket.

No database and no third-party packages: Python standard library only, with a
plain CSV file (sent_log.csv) as the resume log. Secrets come from the
environment, never from files:

    export GMAIL_ADDRESS="you@gmail.com"
    export GMAIL_APP_PASSWORD="your 16-char app password"
"""

import argparse
import csv
import json
import os
import random
import re
import smtplib
import sys
import time
from datetime import datetime
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 465

EMAIL_FIELD = "email"
LOG_HEADER = ("email", "timestamp", "status")

# Template placeholders look like {{field}} (surrounding spaces tolerated). The
# {{...}} syntax is used over str.format/Template because fundraising copy
# contains literal $ and {} that those would choke on. Field names allow letters,
# digits, underscore, dot, and hyphen.
PLACEHOLDER_RE = re.compile(r"{{\s*([\w.\-]+)\s*}}")

# Stop retrying an address after this many failed attempts across all runs, so a
# permanently-bad inbox is not re-hammered every run (a spam signal). A send that
# Gmail refuses outright (bad recipient/sender) is logged as failed_permanent and
# never retried at all.
MAX_SEND_ATTEMPTS = 3

DEFAULT_CONFIG = {
    "from_name": "Club Name",
    "min_delay_seconds": 30,
    "max_delay_seconds": 90,
    "daily_cap": 400,
    "recipients_path": "recipients.csv",
    "template_path": "template.txt",
    "sent_log_path": "sent_log.csv",
}


def die(message):
    """Print a one-line error to stderr and exit non-zero."""
    print("error: " + message, file=sys.stderr)
    sys.exit(1)


# --------------------------------------------------------------------------- #
# Loading: config, secrets, template, recipients
# --------------------------------------------------------------------------- #

def load_config(path):
    """Load config.json, falling back to defaults for any missing keys."""
    config = dict(DEFAULT_CONFIG)
    try:
        with open(path, "r", encoding="utf-8") as f:
            config.update(json.load(f))
    except FileNotFoundError:
        die("config file not found: {} (copy config.json or pass --config)".format(path))
    except json.JSONDecodeError as exc:
        die("config file is not valid JSON ({}): {}".format(path, exc))
    return config


def load_secrets():
    """Read Gmail credentials from the environment. Both are required."""
    address = os.environ.get("GMAIL_ADDRESS")
    password = os.environ.get("GMAIL_APP_PASSWORD")
    if not address:
        die("GMAIL_ADDRESS is not set — export it before running.")
    if not password:
        die("GMAIL_APP_PASSWORD is not set — export your Gmail App Password before running.")
    return address, password


def parse_template(path):
    """First non-blank line is the subject; everything after it is the body."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()
    except FileNotFoundError:
        die("template file not found: {}".format(path))

    subject = None
    body_start = 0
    for i, line in enumerate(lines):
        if line.strip():
            subject = line.strip()
            body_start = i + 1
            break

    if subject is None:
        die("template is empty: {} (need a subject line + body)".format(path))

    # Drop a single blank separator line between subject and body, if present.
    if body_start < len(lines) and not lines[body_start].strip():
        body_start += 1

    body = "\n".join(lines[body_start:]).strip("\n")
    if not body:
        die("template has no body after the subject line: {}".format(path))
    return subject, body


def valid_email(addr):
    """Cheap sanity check: exactly one @, non-empty local part, dotted domain.

    Not RFC-perfect — just enough to skip obviously-broken addresses before we
    waste a send attempt on them (and then retry that failure every run).
    """
    if not addr or " " in addr or addr.count("@") != 1:
        return False
    local, domain = addr.split("@")
    return bool(local) and "." in domain and not domain.startswith(".") \
        and not domain.endswith(".")


def iter_recipients(path, required_fields, stats):
    """Yield validated recipient dicts; report and count malformed rows.

    Streams the CSV (never loads it whole) and keeps every column so the template
    can reference any of them as {{column}}. required_fields (the email column
    plus whatever placeholders the template uses) must all be present in the
    header and non-empty per row, so we never send "Hi {{name}}" or burn an
    attempt on a junk address. Skipped rows increment stats["skipped_invalid"].
    """
    try:
        f = open(path, "r", encoding="utf-8", newline="")
    except FileNotFoundError:
        die("recipients file not found: {}".format(path))

    with f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        missing = [c for c in sorted(required_fields) if c not in header]
        if missing:
            die("recipients CSV missing column(s): {} (template/email need: {})".format(
                ", ".join(missing), ", ".join(sorted(required_fields))))

        for line_no, row in enumerate(reader, start=2):  # line 1 is the header
            # Keep all columns; guard against DictReader's None restkey/restval.
            values = {k: (v or "").strip() for k, v in row.items() if k is not None}
            empty = [field for field in sorted(required_fields) if not values.get(field)]
            if empty:
                print("skip: row {} missing {}".format(line_no, ", ".join(empty)),
                      file=sys.stderr)
                stats["skipped_invalid"] += 1
                continue
            if not valid_email(values.get(EMAIL_FIELD, "")):
                print("skip: row {} invalid email {!r}".format(
                    line_no, values.get(EMAIL_FIELD, "")), file=sys.stderr)
                stats["skipped_invalid"] += 1
                continue
            yield values


# --------------------------------------------------------------------------- #
# Merge + message building
# --------------------------------------------------------------------------- #

def template_fields(*texts):
    """Set of placeholder names ({{field}}) referenced across the template text(s)."""
    fields = set()
    for text in texts:
        fields.update(PLACEHOLDER_RE.findall(text))
    return fields


def merge(text, recipient):
    """Fill {{field}} placeholders from the recipient row. Pure, no I/O.

    Single-pass substitution so a value that itself contains {{...}} is never
    re-expanded.
    """
    return PLACEHOLDER_RE.sub(lambda m: recipient.get(m.group(1), ""), text)


def build_message(subject, body, recipient, from_name, from_address):
    """Build a plain-text EmailMessage for one merged recipient."""
    msg = EmailMessage()
    msg["From"] = "{} <{}>".format(from_name, from_address)
    msg["To"] = recipient["email"]
    msg["Subject"] = merge(subject, recipient)
    msg["Date"] = formatdate(localtime=True)
    msg["Message-ID"] = make_msgid(domain=from_address.rsplit("@", 1)[-1])
    msg.set_content(merge(body, recipient))
    return msg


# --------------------------------------------------------------------------- #
# Persistent SMTP sender with auto-reconnect
# --------------------------------------------------------------------------- #

class SmtpSender:
    """One login, reused for the whole batch, reconnecting if Gmail drops us."""

    def __init__(self, address, password):
        self.address = address
        self.password = password
        self.server = None

    def connect(self):
        self.server = smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT)
        self.server.login(self.address, self.password)

    def send(self, msg):
        """Send one message, reconnecting once if the connection was dropped."""
        if self.server is None:
            self.connect()
        try:
            self.server.send_message(msg)
        except smtplib.SMTPServerDisconnected:
            # Gmail closes idle connections during long pauses — reconnect + retry once.
            self.connect()
            self.server.send_message(msg)

    def close(self):
        if self.server is not None:
            try:
                self.server.quit()
            except smtplib.SMTPException:
                pass
            self.server = None


def open_sender(from_address, password):
    """Create + log in a sender, failing with a clear message on auth errors."""
    sender = SmtpSender(from_address, password)
    try:
        sender.connect()
    except smtplib.SMTPAuthenticationError:
        die("Gmail rejected the login — check GMAIL_ADDRESS and that "
            "GMAIL_APP_PASSWORD is a valid App Password (not your normal password).")
    except (smtplib.SMTPException, OSError) as exc:
        die("could not connect to Gmail SMTP: {}".format(exc))
    return sender


# --------------------------------------------------------------------------- #
# Resume log
# --------------------------------------------------------------------------- #

def read_log(path, today_str):
    """Return (already_sent, sent_today, do_not_retry) from the send log.

    O(1) membership lookups on re-run; a missing log just means nothing sent yet.
    Only rows with status "sent" count toward already_sent / the daily cap.

    do_not_retry holds addresses we should stop attempting: anything Gmail
    refused outright (status "failed_permanent:...") plus any address that has
    racked up >= MAX_SEND_ATTEMPTS transient failures. Without this, a bad
    address is re-sent on every run, which both wastes the daily cap and looks
    like spam to Gmail.
    """
    already_sent = set()
    sent_today = 0
    permanent_fail = set()
    failed_counts = {}
    try:
        with open(path, "r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 3:
                    continue
                email, timestamp, status = row[0], row[1], row[2]
                if status == "sent":
                    already_sent.add(email)
                    if timestamp.startswith(today_str):
                        sent_today += 1
                elif status.startswith("failed_permanent"):
                    permanent_fail.add(email)
                elif status.startswith("failed"):
                    failed_counts[email] = failed_counts.get(email, 0) + 1
    except FileNotFoundError:
        pass

    do_not_retry = set(permanent_fail)
    do_not_retry.update(
        email for email, n in failed_counts.items() if n >= MAX_SEND_ATTEMPTS)
    return already_sent, sent_today, do_not_retry


def open_log(path):
    """Open the send log for append, writing a header if the file is new."""
    is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", encoding="utf-8", newline="")
    writer = csv.writer(f)
    if is_new:
        writer.writerow(LOG_HEADER)
        f.flush()
    return f, writer


def log_attempt(log_file, writer, email, status):
    """Append one attempt and flush so a crash never loses state."""
    writer.writerow([email, datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), status])
    log_file.flush()


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #

def render(msg):
    """Pretty-print a built message for --dry-run / --test preview."""
    print("-" * 60)
    print("From:    " + msg["From"])
    print("To:      " + msg["To"])
    print("Subject: " + msg["Subject"])
    print()
    print(msg.get_content().rstrip("\n"))
    print("-" * 60)


def print_summary(stats, log_path):
    skipped = (stats["skipped_sent"] + stats["skipped_invalid"]
               + stats["skipped_failed"])
    print("\nsummary: {} sent, {} failed, {} skipped "
          "({} already-sent, {} invalid rows, {} gave-up after retries)".format(
              stats["sent"], stats["failed"], skipped,
              stats["skipped_sent"], stats["skipped_invalid"],
              stats["skipped_failed"]))
    print("log: {}".format(os.path.abspath(log_path)))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #

def cmd_dry_run(subject, body, config, from_address, required_fields, limit):
    """Render merged emails to the terminal (up to --limit). Sends nothing."""
    stats = {"skipped_invalid": 0}
    count = 0
    for recipient in iter_recipients(config["recipients_path"], required_fields, stats):
        if limit is not None and count >= limit:
            break
        render(build_message(subject, body, recipient, config["from_name"], from_address))
        count += 1
    print("\ndry-run: {} email(s) rendered, none sent "
          "({} invalid row(s) skipped).".format(count, stats["skipped_invalid"]))


def cmd_test(subject, body, config, from_address, password, required_fields):
    """Send ONE merged email to yourself, then exit."""
    # Fill every placeholder the template uses so nothing renders blank.
    recipient = {field: "Test " + field for field in required_fields}
    recipient[EMAIL_FIELD] = from_address
    msg = build_message(subject, body, recipient, config["from_name"], from_address)
    print("Sending one test email to {} ...".format(from_address))
    render(msg)
    sender = open_sender(from_address, password)
    try:
        sender.send(msg)
    except smtplib.SMTPException as exc:
        sender.close()
        die("test send failed: {}".format(exc))
    sender.close()
    print("\ntest: sent to {}. Check your inbox (and confirm it is NOT in spam).".format(
        from_address))


def cmd_send(subject, body, config, from_address, password, required_fields, limit):
    """Real batch send: paced, capped, resumable, fault-isolated."""
    log_path = config["sent_log_path"]
    today = datetime.now().strftime("%Y-%m-%d")
    already_sent, sent_today, do_not_retry = read_log(log_path, today)
    daily_cap = config["daily_cap"]
    min_delay = config["min_delay_seconds"]
    max_delay = config["max_delay_seconds"]

    stats = {"sent": 0, "failed": 0, "skipped_sent": 0,
             "skipped_invalid": 0, "skipped_failed": 0}

    if sent_today >= daily_cap:
        print("Daily cap already reached ({}/{} sent today). Resume tomorrow.".format(
            sent_today, daily_cap))
        return

    sender = open_sender(from_address, password)
    log_file, writer = open_log(log_path)
    sent_this_run = 0

    try:
        for recipient in iter_recipients(config["recipients_path"], required_fields, stats):
            email = recipient["email"]

            if email in already_sent:
                stats["skipped_sent"] += 1
                continue
            if email in do_not_retry:
                stats["skipped_failed"] += 1
                continue
            if sent_today >= daily_cap:
                print("Daily cap reached ({}/{}). Stopping — resume tomorrow.".format(
                    sent_today, daily_cap))
                break
            if limit is not None and sent_this_run >= limit:
                print("Run limit reached (--limit {}). Stopping.".format(limit))
                break

            # Pace between sends (not before the first one this run).
            if sent_this_run > 0:
                delay = random.uniform(min_delay, max_delay)
                print("  waiting {:.0f}s ...".format(delay))
                time.sleep(delay)

            msg = build_message(subject, body, recipient, config["from_name"], from_address)
            try:
                sender.send(msg)
            except (smtplib.SMTPRecipientsRefused, smtplib.SMTPSenderRefused) as exc:
                # Gmail refused the address outright — retrying never helps and
                # repeatedly mailing a dead address hurts sender reputation.
                reason = str(exc).replace("\n", " ")[:120]
                log_attempt(log_file, writer, email, "failed_permanent:" + reason)
                do_not_retry.add(email)
                print("failed (permanent, won't retry): {} ({})".format(email, reason),
                      file=sys.stderr)
                stats["failed"] += 1
                continue
            except smtplib.SMTPException as exc:
                # Transient (timeout, temporary 4xx, etc.) — retried on the next
                # run, but only up to MAX_SEND_ATTEMPTS before we give up.
                reason = str(exc).replace("\n", " ")[:120]
                log_attempt(log_file, writer, email, "failed:" + reason)
                print("failed: {} ({})".format(email, reason), file=sys.stderr)
                stats["failed"] += 1
                continue

            log_attempt(log_file, writer, email, "sent")
            already_sent.add(email)
            sent_today += 1
            sent_this_run += 1
            stats["sent"] += 1
            print("sent: {}  [{}/{} today]".format(email, sent_today, daily_cap))

    except KeyboardInterrupt:
        print("\nInterrupted — stopping cleanly. Log is intact; re-run to resume.")
    finally:
        log_file.close()
        sender.close()
        print_summary(stats, log_path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Send personalized cold emails from Gmail via CSV mail merge.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="config.json",
                        help="path to config JSON (default: config.json)")
    parser.add_argument("--recipients",
                        help="override recipients CSV path from config")
    parser.add_argument("--template",
                        help="override template path from config")
    parser.add_argument("--limit", type=int, metavar="N",
                        help="send (or render) at most N emails this run")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true",
                      help="render merged emails to the terminal, send nothing")
    mode.add_argument("--test", action="store_true",
                      help="send one merged email to your own GMAIL_ADDRESS, then exit")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.limit is not None and args.limit < 1:
        die("--limit must be a positive integer.")

    config = load_config(args.config)
    if args.recipients:
        config["recipients_path"] = args.recipients
    if args.template:
        config["template_path"] = args.template

    subject, body = parse_template(config["template_path"])
    required_fields = template_fields(subject, body) | {EMAIL_FIELD}
    unfilled = "[FILL" in subject or "[FILL" in body

    if args.dry_run:
        from_address = os.environ.get("GMAIL_ADDRESS", "you@gmail.com")
        cmd_dry_run(subject, body, config, from_address, required_fields, args.limit)
        if unfilled:
            print("\nWARNING: template still has [FILL ...] placeholders "
                  "(sender name, opt-out address). Replace them before a real send.",
                  file=sys.stderr)
        return

    if unfilled:
        die("template still has [FILL ...] placeholders (sender name, opt-out "
            "address) — replace them in template.txt before sending. These are "
            "required for CAN-SPAM compliance.")

    from_address, password = load_secrets()
    if args.test:
        cmd_test(subject, body, config, from_address, password, required_fields)
    else:
        cmd_send(subject, body, config, from_address, password, required_fields, args.limit)


if __name__ == "__main__":
    main()
