"""wishlist — Dom names one item, one Claude web-search call checks its status.

Sibling of `releases` but user-driven instead of the fixed eight categories:
no persisted list, no schedule — a pure one-shot query, same shape as
`places`/`projects`. Good for "is the PS5 Pro restocked" / "did that jacket
go on sale" one-off checks.

Run: python -m skills.wishlist --input '{"item":"PS5 Pro restock"}'
"""

import argparse
import json
import re

from .._lib import claude, envelope

SKILL_ID = "wishlist"


def _build_prompt(item):
    return (
        "Search the web for the current status of this item Dom is watching "
        f'for:\n\n"{item}"\n\n'
        "Figure out whether it's currently available/in stock/on sale/"
        "released, or still pending, and any relevant price. Return ONLY "
        "strict JSON, no markdown fences, no commentary, matching this "
        "schema:\n"
        '{"status": "...", "detail": "...", "price": "...", '
        '"source_url": "...", "confidence": "..."}\n\n'
        "Rules:\n"
        "- status: a short label, e.g. \"in stock\", \"sold out\", "
        "\"on sale\", \"not yet released\", \"unknown\".\n"
        "- detail: 1-2 sentences of the actual current situation.\n"
        "- price: the current price if known, else \"\".\n"
        "- source_url: a real URL backing the claim, else \"\".\n"
        "- confidence: one of \"confirmed\", \"rumored\", \"unconfirmed\".\n"
    )


def _extract_json(raw_result):
    text = raw_result.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def run(item):
    envelope.progress(f"wishlist: checking status of {item!r}")
    try:
        response = claude.ask(_build_prompt(item))
        parsed = _extract_json(response["result"])
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
        envelope.progress(f"wishlist: failed - {exc}")
        return envelope.build(
            SKILL_ID,
            ok=False,
            summary=f"Failed to check status of '{item}'",
            sections=[],
            error=str(exc),
        )

    status = str(parsed.get("status", "")).strip() or "unknown"
    detail = str(parsed.get("detail", "")).strip()
    price = str(parsed.get("price", "")).strip()
    source_url = str(parsed.get("source_url", "")).strip()
    confidence = str(parsed.get("confidence", "")).strip() or "unconfirmed"

    sections = [
        {
            "type": "stat",
            "title": "At a glance",
            "stats": [
                {"label": "Status", "value": status},
                {"label": "Confidence", "value": confidence},
            ]
            + ([{"label": "Price", "value": price}] if price else []),
        }
    ]
    if detail or source_url:
        body = detail
        if source_url:
            body += f"\n\nSource: {source_url}"
        sections.append({"type": "text", "title": item, "body": body.strip()})

    summary = f"'{item}': {status}" + (f" ({price})" if price else "")
    envelope.progress("wishlist: done")
    return envelope.build(SKILL_ID, ok=True, summary=summary, sections=sections)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="{}")
    args = parser.parse_args()

    try:
        payload = json.loads(args.input)
    except json.JSONDecodeError as e:
        env = envelope.build(SKILL_ID, ok=False, summary="Invalid --input JSON", sections=[], error=str(e))
        envelope.validate(env)
        print(json.dumps(env))
        return

    item = (payload.get("item") or "").strip()
    if not item:
        env = envelope.build(
            SKILL_ID,
            ok=False,
            summary="No item provided",
            sections=[],
            error='Input must include a non-empty "item" string, e.g. {"item": "PS5 Pro restock"}.',
        )
        envelope.validate(env)
        print(json.dumps(env))
        return

    try:
        env = run(item)
    except Exception as e:  # noqa: BLE001 - skill must always emit a valid envelope, never crash bare
        env = envelope.build(
            SKILL_ID, ok=False, summary="wishlist failed", sections=[], error=f"{type(e).__name__}: {e}"
        )

    envelope.validate(env)
    print(json.dumps(env))


if __name__ == "__main__":
    main()
