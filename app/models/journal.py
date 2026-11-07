from app.db import fetchall_dict, fetchone_dict


def get_recent_entries(limit=4):
    return fetchall_dict("""
        SELECT id, entry_date, entry_type, tier_number, written_text, created_at
        FROM journal_entries
        ORDER BY created_at DESC
        LIMIT %s
    """, (limit,))


def get_all_entries_full():
    """Every journal entry with full snapshots + comments, oldest first.
    Used to compile one combined document for Drive upload."""
    entries = fetchall_dict("""
        SELECT id, session_id, entry_date, entry_type, tier_number,
               auto_data_snapshot, interpretations_snapshot,
               portfolio_entries_snapshot, written_text, created_at
        FROM journal_entries
        ORDER BY entry_date ASC, created_at ASC
    """)
    comments = fetchall_dict("""
        SELECT entry_id, comment_text, created_at
        FROM journal_comments
        ORDER BY created_at ASC
    """)
    by_entry = {}
    for c in comments:
        by_entry.setdefault(c["entry_id"], []).append(c)
    for e in entries:
        e["comments"] = by_entry.get(e["id"], [])
    return entries


def get_entry_with_comments(entry_id):
    entry = fetchone_dict("""
        SELECT id, session_id, entry_date, entry_type, tier_number,
               auto_data_snapshot, interpretations_snapshot,
               portfolio_entries_snapshot, written_text, created_at
        FROM journal_entries
        WHERE id = %s
    """, (entry_id,))
    if not entry:
        return None, []
    comments = fetchall_dict("""
        SELECT id, comment_text, created_at
        FROM journal_comments
        WHERE entry_id = %s
        ORDER BY created_at
    """, (entry_id,))
    return entry, comments
