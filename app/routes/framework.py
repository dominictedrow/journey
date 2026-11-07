import json

from flask import Blueprint, render_template, redirect, url_for, request

from app.db import fetchone_dict, fetchall_dict, execute, execute_returning
from app.models.market_data import get_latest, get_latest_by_tier, get_series_history, get_multi_series_history

bp = Blueprint("framework", __name__)

TIER_NAMES = {
    1: "Macro Regime Classification",
    2: "Country & Region / Crypto Regime",
    3: "Sector & Asset-Class Rotation",
    4: "Individual Selection",
    5: "Risk Management & Monitoring",
}

TIER_DESCRIPTIONS = {
    1: "Classify the current macro regime across four quadrants: Goldilocks, Reflation, Stagflation, or Deflation.",
    2: "Score countries 1–5 across four dimensions; assess the crypto cycle phase and BTC dominance.",
    3: "Determine sector overweights based on business cycle position and commodity overlays.",
    4: "Apply the 4-factor checklist (Quality, Valuation, Momentum, Catalyst) to individual candidates.",
    5: "Review position sizing, set stop levels, and check regime-change triggers.",
}

TIER_NEWS_CATS = {
    1: ("macro",),
    2: ("macro", "crypto"),
    3: ("macro",),
    4: ("macro",),
    5: ("macro", "crypto"),
}


def _active_session():
    return fetchone_dict("""
        SELECT id, last_active_tier, completed_tiers, started_at
        FROM framework_sessions
        WHERE is_active = TRUE AND is_complete = FALSE
          AND session_type = 'full_framework'
        ORDER BY started_at DESC LIMIT 1
    """)


def _tier_series(tier_num):
    series = get_latest_by_tier(tier_num)
    ids = [s["series_id"] for s in series]
    histories = get_multi_series_history(ids, 30)
    for s in series:
        hist = histories.get(s["series_id"], [])
        s["chart_data"] = [
            {"x": r["recorded_at"].isoformat(), "y": float(r["value"])}
            for r in hist
        ]
    return series


def _tier_news(tier_num, limit=14):
    cats = TIER_NEWS_CATS.get(tier_num, ("macro",))
    placeholders = ",".join(["%s"] * len(cats))
    return fetchall_dict(f"""
        SELECT id, source_name, title, url, published_at, category
        FROM news_items
        WHERE is_active = TRUE AND category IN ({placeholders})
        ORDER BY published_at DESC LIMIT %s
    """, (*cats, limit))


def _existing_interps(session_id, tier_num):
    rows = fetchall_dict("""
        SELECT data_item_key, interpretation
        FROM tier_interpretations
        WHERE session_id = %s AND tier = %s
    """, (session_id, tier_num))
    return {r["data_item_key"]: r["interpretation"] for r in rows}


def _save_interps(session_id, tier_num, series, form):
    execute("DELETE FROM tier_interpretations WHERE session_id = %s AND tier = %s",
            (session_id, tier_num))
    for s in series:
        key = s["series_id"]
        text = (form.get(f"interp_{key}") or "").strip()
        if text:
            execute("""
                INSERT INTO tier_interpretations
                  (session_id, tier, data_item_key, data_item_label, interpretation)
                VALUES (%s, %s, %s, %s, %s)
            """, (session_id, tier_num, key, s["metric_name"] or key, text))


def _save_portfolio_entries(session_id, form):
    execute("DELETE FROM session_portfolio_entries WHERE session_id = %s", (session_id,))
    tickers = form.getlist("pe_ticker")
    prices  = form.getlist("pe_price")
    qtys    = form.getlist("pe_qty")
    types   = form.getlist("pe_type")
    for ticker, price, qty, atype in zip(tickers, prices, qtys, types):
        ticker = ticker.strip().upper()
        if not ticker:
            continue
        try:
            p = float(price) if price.strip() else None
            q = float(qty)   if qty.strip()   else None
        except (ValueError, AttributeError):
            continue
        execute("""
            INSERT INTO session_portfolio_entries
              (session_id, ticker, entry_price, quantity, asset_type)
            VALUES (%s, %s, %s, %s, %s)
        """, (session_id, ticker, p, q, atype or "stock"))


def _create_tier_journal_entry(session_id, tier_num, written_text):
    """Build tier-scoped snapshots from DB and insert a journal_entries row. Returns new entry id."""
    tier_series = get_latest_by_tier(tier_num)
    auto_snap = {
        s["series_id"]: {
            "value": float(s["value"]) if s["value"] is not None else None,
            "text_value": s["text_value"],
            "label": s["metric_name"],
            "category": s["category"],
            "tier": s["tier"],
        }
        for s in tier_series
    }

    interps = fetchall_dict("""
        SELECT data_item_key, data_item_label, interpretation
        FROM tier_interpretations WHERE session_id = %s AND tier = %s
    """, (session_id, tier_num))
    interp_snap = {str(tier_num): [
        {"key": r["data_item_key"], "label": r["data_item_label"], "interpretation": r["interpretation"]}
        for r in interps
    ]}

    pe_rows = fetchall_dict("""
        SELECT ticker, entry_price, quantity, asset_type
        FROM session_portfolio_entries WHERE session_id = %s
    """, (session_id,))
    pe_snap = [
        {
            "ticker": r["ticker"],
            "entry_price": float(r["entry_price"]) if r["entry_price"] else None,
            "quantity": float(r["quantity"]) if r["quantity"] else None,
            "asset_type": r["asset_type"],
        }
        for r in pe_rows
    ]

    row = execute_returning("""
        INSERT INTO journal_entries
          (session_id, entry_type, tier_number, auto_data_snapshot,
           interpretations_snapshot, portfolio_entries_snapshot, written_text)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (session_id, f"tier_{tier_num}", tier_num,
          json.dumps(auto_snap), json.dumps(interp_snap), json.dumps(pe_snap),
          written_text or None))
    return row[0]


# ── Routes ────────────────────────────────────────────────────────────────────

@bp.route("/framework/")
def start():
    session = _active_session()
    return render_template("framework/start.html", active_session=session)


@bp.route("/framework/new")
def new_session():
    execute("""
        UPDATE framework_sessions
        SET is_active = FALSE
        WHERE is_active = TRUE AND is_complete = FALSE AND session_type = 'full_framework'
    """)
    execute("""
        INSERT INTO framework_sessions (session_type, last_active_tier)
        VALUES ('full_framework', 1)
    """)
    return redirect(url_for("framework.tier", n=1))


@bp.route("/framework/tier/<int:n>", methods=["GET", "POST"])
def tier(n):
    if not (1 <= n <= 5):
        return redirect(url_for("framework.start"))

    session = _active_session()
    if not session:
        return redirect(url_for("framework.start"))
    session_id = session["id"]

    series = _tier_series(n)

    if request.method == "POST":
        action = request.form.get("action", "next")
        _save_interps(session_id, n, series, request.form)
        _save_portfolio_entries(session_id, request.form)

        completed = list(session["completed_tiers"] or [])
        if n not in completed:
            completed.append(n)

        if action == "prev":
            new_tier = max(1, n - 1)
        elif action == "exit":
            new_tier = n
        else:
            new_tier = min(5, n + 1)
        execute("""
            UPDATE framework_sessions
            SET last_active_tier = %s, completed_tiers = %s
            WHERE id = %s
        """, (new_tier, completed, session_id))

        if action == "exit":
            return redirect(url_for("home.index"))
        elif action == "prev":
            return redirect(url_for("framework.tier", n=max(1, n - 1)))
        elif n < 5:
            return redirect(url_for("framework.tier", n=n + 1))
        else:
            return redirect(url_for("framework.complete", session_id=session_id))

    existing = _existing_interps(session_id, n)
    news = _tier_news(n)
    pe_rows = fetchall_dict("""
        SELECT ticker, entry_price, quantity, asset_type
        FROM session_portfolio_entries WHERE session_id = %s
    """, (session_id,))

    return render_template(
        "framework/tier.html",
        tier_num=n,
        tier_name=TIER_NAMES[n],
        tier_desc=TIER_DESCRIPTIONS[n],
        series=series,
        news=news,
        session=session,
        existing=existing,
        completed_tiers=list(session["completed_tiers"] or []),
        pe_rows=pe_rows,
        tier_names=TIER_NAMES,
    )


@bp.route("/framework/complete/<int:session_id>", methods=["GET", "POST"])
def complete(session_id):
    session = fetchone_dict("""
        SELECT id, session_type, started_at, completed_tiers
        FROM framework_sessions WHERE id = %s
    """, (session_id,))
    if not session:
        return redirect(url_for("framework.start"))

    all_interps = fetchall_dict("""
        SELECT tier, data_item_key, data_item_label, interpretation
        FROM tier_interpretations
        WHERE session_id = %s ORDER BY tier, id
    """, (session_id,))

    pe_rows = fetchall_dict("""
        SELECT ticker, entry_price, quantity, asset_type
        FROM session_portfolio_entries WHERE session_id = %s
    """, (session_id,))

    if request.method == "POST":
        written = (request.form.get("written_text") or "").strip()

        latest = get_latest()
        auto_snap = {
            k: {
                "value": float(v["value"]) if v["value"] is not None else None,
                "text_value": v["text_value"],
                "label": v["metric_name"],
                "category": v["category"],
                "tier": v["tier"],
            }
            for k, v in latest.items()
        }
        interp_snap = {}
        for row in all_interps:
            interp_snap.setdefault(str(row["tier"]), []).append({
                "key": row["data_item_key"],
                "label": row["data_item_label"],
                "interpretation": row["interpretation"],
            })
        pe_snap = [
            {
                "ticker": r["ticker"],
                "entry_price": float(r["entry_price"]) if r["entry_price"] else None,
                "quantity": float(r["quantity"]) if r["quantity"] else None,
                "asset_type": r["asset_type"],
            }
            for r in pe_rows
        ]

        row = execute_returning("""
            INSERT INTO journal_entries
              (session_id, entry_type, auto_data_snapshot, interpretations_snapshot,
               portfolio_entries_snapshot, written_text)
            VALUES (%s, 'full_framework', %s, %s, %s, %s)
            RETURNING id
        """, (session_id, json.dumps(auto_snap), json.dumps(interp_snap),
               json.dumps(pe_snap), written or None))
        journal_id = row[0]

        execute("""
            UPDATE framework_sessions
            SET is_complete = TRUE, is_active = FALSE, completed_at = NOW()
            WHERE id = %s
        """, (session_id,))

        return redirect(url_for("journal.entry", entry_id=journal_id))

    grouped = {}
    for row in all_interps:
        grouped.setdefault(row["tier"], []).append(row)

    return render_template(
        "framework/complete.html",
        session=session,
        grouped_interps=grouped,
        pe_rows=pe_rows,
        tier_names=TIER_NAMES,
    )


@bp.route("/framework/exit/<int:session_id>", methods=["POST"])
def exit_session(session_id):
    tier_num = request.form.get("current_tier", type=int, default=1)
    execute("UPDATE framework_sessions SET last_active_tier = %s WHERE id = %s",
            (tier_num, session_id))
    return redirect(url_for("home.index"))


# ── Standalone tier (Phase 5) ─────────────────────────────────────────────────

@bp.route("/tiers/<int:n>", methods=["GET", "POST"])
def tier_standalone(n):
    if not (1 <= n <= 5):
        return redirect(url_for("home.index"))
    series = _tier_series(n)
    news = _tier_news(n)

    if request.method == "POST":
        written = (request.form.get("written_text") or "").strip()

        sess_row = execute_returning("""
            INSERT INTO framework_sessions
              (session_type, last_active_tier, completed_tiers)
            VALUES (%s, %s, %s) RETURNING id
        """, (f"tier_{n}", n, [n]))
        session_id = sess_row[0]

        _save_interps(session_id, n, series, request.form)
        _save_portfolio_entries(session_id, request.form)

        journal_id = _create_tier_journal_entry(session_id, n, written)

        execute("""
            UPDATE framework_sessions
            SET is_complete = TRUE, is_active = FALSE, completed_at = NOW()
            WHERE id = %s
        """, (session_id,))

        return redirect(url_for("journal.entry", entry_id=journal_id))

    return render_template(
        "framework/tier.html",
        tier_num=n,
        tier_name=TIER_NAMES[n],
        tier_desc=TIER_DESCRIPTIONS[n],
        series=series,
        news=news,
        session=None,
        existing={},
        completed_tiers=[],
        pe_rows=[],
        tier_names=TIER_NAMES,
        standalone=True,
    )
