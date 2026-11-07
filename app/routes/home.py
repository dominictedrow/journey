import json
from datetime import datetime, timezone

from flask import Blueprint, render_template

from app.db import fetchone_dict, fetchall_dict
from app.models.market_data import get_latest, get_biggest_shifts
from app.models.journal import get_recent_entries
from config.settings import resource_path

bp = Blueprint("home", __name__)

_SOURCES_PATH = resource_path("config/sources.json")

def _load_label_map():
    with open(_SOURCES_PATH) as f:
        sources = json.load(f)
    return {s['id']: s['label'] for s in sources['series']}

_LABEL_MAP = _load_label_map()


def _greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    return "Good evening"


@bp.route("/")
def index():
    latest = get_latest()

    active_session = fetchone_dict("""
        SELECT id, session_type, started_at, last_active_tier, completed_tiers
        FROM framework_sessions
        WHERE is_active = TRUE AND is_complete = FALSE
        ORDER BY started_at DESC LIMIT 1
    """)

    tier_activity = {}
    for row in fetchall_dict("""
        SELECT tier_number, MAX(entry_date) AS last_date
        FROM journal_entries
        WHERE tier_number IS NOT NULL
        GROUP BY tier_number
    """):
        tier_activity[row['tier_number']] = row['last_date']

    stories = fetchall_dict("""
        SELECT id, source_name, title, url, summary, published_at, category
        FROM news_items
        WHERE is_active = TRUE
        ORDER BY engagement_score DESC, published_at DESC
        LIMIT 6
    """)

    shifts = get_biggest_shifts(limit=8)
    for s in shifts:
        s['label'] = _LABEL_MAP.get(s['series_id'], s['series_id'])

    recent_journal = get_recent_entries(limit=4)

    return render_template(
        "home/index.html",
        greeting=_greeting(),
        now=datetime.now(timezone.utc),
        latest=latest,
        active_session=active_session,
        tier_activity=tier_activity,
        stories=stories,
        shifts=shifts,
        recent_journal=recent_journal,
    )
