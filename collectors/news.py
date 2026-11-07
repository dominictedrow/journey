import json
import requests
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from datetime import datetime, timezone
from collectors.base import save_news
from config.settings import resource_path

SOURCES_PATH = resource_path("config/sources.json")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
MAX_PER_FEED = 20


def _parse_date(pub):
    """Parse an RSS pubDate. Handles RFC-822 (most feeds) and ISO-8601
    (e.g. Yahoo Finance: 2026-06-22T18:47:00Z). Returns None on failure;
    callers store NULL rather than skipping the item."""
    if not pub:
        return None
    pub = pub.strip()
    dt = None
    try:
        dt = parsedate_to_datetime(pub)
    except Exception:
        dt = None
    if dt is None:
        try:
            dt = datetime.fromisoformat(pub.replace("Z", "+00:00"))
        except Exception:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _parse_rss(url, source_name, category, tier):
    items = []
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            channel = root

        for item in channel.findall("item")[:MAX_PER_FEED]:
            title = (item.findtext("title") or "").strip()
            link  = (item.findtext("link") or "").strip()
            desc  = (item.findtext("description") or "").strip()
            pub   = item.findtext("pubDate") or ""

            if not title or not link:
                continue

            published_at = _parse_date(pub)

            items.append({
                "source_name":  source_name,
                "title":        title[:500],
                "url":          link[:1000],
                "summary":      desc[:1000] if desc else None,
                "published_at": published_at,
                "category":     category,
                "tier":         tier,
            })
    except Exception as e:
        print(f"news: error fetching {source_name} ({url}): {e}")

    return items


def collect():
    with open(SOURCES_PATH) as f:
        config = json.load(f)

    all_items = []
    for feed in config.get("rss_feeds", []):
        items = _parse_rss(feed["url"], feed["source_name"], feed["category"], feed["tier"])
        all_items.extend(items)

    save_news(all_items)
    print(f"news: saved up to {len(all_items)} items (deduped by URL)")


if __name__ == "__main__":
    collect()
