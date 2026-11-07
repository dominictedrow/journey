from datetime import datetime

from flask import Blueprint, render_template, redirect, url_for, request, Response

from app.db import execute, fetchall_dict, fetchone_dict
from app.models.journal import get_entry_with_comments, get_all_entries_full

bp = Blueprint("journal", __name__)

VALID_TYPES = (
    "full_framework", "tier_1", "tier_2", "tier_3", "tier_4", "tier_5",
    "daily", "weekly", "monthly", "portfolio",
)


@bp.route("/journal/")
def list_entries():
    start = request.args.get("start", "")
    end = request.args.get("end", "")
    type_filter = request.args.get("type", "")

    where, params = [], []
    if start:
        where.append("entry_date >= %s")
        params.append(start)
    if end:
        where.append("entry_date <= %s")
        params.append(end)
    if type_filter in VALID_TYPES:
        where.append("entry_type = %s")
        params.append(type_filter)

    where_sql = ("WHERE " + " AND ".join(where)) if where else ""
    rows = fetchall_dict(f"""
        SELECT id, entry_date, entry_type, tier_number, written_text, created_at
        FROM journal_entries
        {where_sql}
        ORDER BY created_at DESC
        LIMIT 300
    """, params)

    grouped = {}
    for row in rows:
        key = row["entry_date"].strftime("%B %Y")
        grouped.setdefault(key, []).append(row)

    return render_template(
        "journal/list.html",
        grouped=grouped,
        start=start, end=end, type_filter=type_filter,
        valid_types=VALID_TYPES,
    )


def _fmt_num(v):
    if v is None:
        return "—"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if f == int(f):
        return f"{int(f):,}"
    return f"{f:,.4f}".rstrip("0").rstrip(".")


def _entry_heading(e):
    if e["entry_type"] == "full_framework":
        return "Full Framework"
    if e["entry_type"].startswith("tier_"):
        return f"Tier {e['tier_number']}"
    return e["entry_type"].replace("_", " ").title()


def _entry_to_markdown(e):
    lines = []
    lines.append(f"## {e['entry_date'].strftime('%A, %B %-d, %Y')} — {_entry_heading(e)}")
    lines.append("")

    auto = e.get("auto_data_snapshot")
    if auto:
        lines.append("### Market Data Snapshot")
        lines.append("")
        lines.append("| Series | Value | Category | Tier |")
        lines.append("| --- | --- | --- | --- |")
        for sid, s in auto.items():
            s = s or {}
            label = s.get("label") or sid
            if s.get("value") is not None:
                val = _fmt_num(s.get("value"))
                if s.get("text_value"):
                    val = f"{val} ({s['text_value']})"
            elif s.get("text_value"):
                val = s["text_value"]
            else:
                val = "—"
            lines.append(f"| {label} | {val} | {s.get('category') or '—'} | {s.get('tier') or '—'} |")
        lines.append("")

    interps = e.get("interpretations_snapshot")
    if interps:
        lines.append("### Interpretations")
        lines.append("")
        for tier_key, items in interps.items():
            if not items:
                continue
            written = [i for i in items if i.get("interpretation")]
            if not written:
                continue
            lines.append(f"**Tier {tier_key}**")
            lines.append("")
            for item in written:
                lines.append(f"- *{item.get('label') or item.get('key')}:* {item['interpretation']}")
            lines.append("")

    pf = e.get("portfolio_entries_snapshot")
    if pf:
        lines.append("### Portfolio Entries")
        lines.append("")
        lines.append("| Ticker | Entry Price | Quantity | Type |")
        lines.append("| --- | --- | --- | --- |")
        for p in pf:
            price = f"${float(p['entry_price']):.2f}" if p.get("entry_price") is not None else "—"
            qty = p["quantity"] if p.get("quantity") is not None else "—"
            lines.append(f"| {p.get('ticker','')} | {price} | {qty} | {p.get('asset_type','')} |")
        lines.append("")

    if e.get("written_text"):
        lines.append("### Notes")
        lines.append("")
        lines.append(e["written_text"].rstrip())
        lines.append("")

    if e.get("comments"):
        lines.append("### Comments")
        lines.append("")
        for c in e["comments"]:
            lines.append(f"- {c['comment_text']}  ")
            lines.append(f"  _{c['created_at'].strftime('%Y-%m-%d %H:%M')}_")
        lines.append("")

    lines.append("---")
    lines.append("")
    return "\n".join(lines)


@bp.route("/journal/upload-to-drive")
def upload_to_drive():
    """Compile every journal entry into one long Markdown document and download it."""
    entries = get_all_entries_full()
    parts = [
        "# Market Research Journal",
        "",
        f"_Combined export of {len(entries)} entr{'y' if len(entries) == 1 else 'ies'} "
        f"generated {datetime.now().strftime('%Y-%m-%d %H:%M')}._",
        "",
        "---",
        "",
    ]
    parts.extend(_entry_to_markdown(e) for e in entries)
    doc = "\n".join(parts)

    filename = f"market_research_journal_{datetime.now().strftime('%Y%m%d')}.md"
    return Response(
        doc,
        mimetype="text/markdown",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@bp.route("/journal/<int:entry_id>")
def entry(entry_id):
    e, comments = get_entry_with_comments(entry_id)
    if not e:
        return redirect(url_for("journal.list_entries"))
    return render_template("journal/entry.html", entry=e, comments=comments)


@bp.route("/journal/<int:entry_id>/export")
def export_entry(entry_id):
    e, comments = get_entry_with_comments(entry_id)
    if not e:
        return redirect(url_for("journal.list_entries"))
    return render_template("journal/entry.html", entry=e, comments=comments, print_mode=True)


@bp.route("/journal/<int:entry_id>/comment", methods=["POST"])
def add_comment(entry_id):
    if not fetchone_dict("SELECT id FROM journal_entries WHERE id = %s", (entry_id,)):
        return redirect(url_for("journal.list_entries"))
    text = (request.form.get("comment_text") or "").strip()
    if text:
        execute(
            "INSERT INTO journal_comments (entry_id, comment_text) VALUES (%s, %s)",
            (entry_id, text),
        )
    return redirect(url_for("journal.entry", entry_id=entry_id))
