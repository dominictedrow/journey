"""AI concept/model -> concrete buildable project ideas.

Run: python -m skills.projects --input '{"concept":"transformer attention"}'
"""

import argparse
import json
import re

from .._lib import claude, envelope

_TABLE_COLUMNS = ["Project/Scope", "Skill Exercised", "Starter Pointer", "Difficulty"]
# Above this length, a scope description reads better as its own text block
# than crammed into a table cell.
_SCOPE_TABLE_LIMIT = 200


def _build_prompt(concept):
    return (
        "You are helping a software engineer turn an AI concept into concrete, "
        "buildable practice projects.\n\n"
        f'Concept: "{concept}"\n\n'
        "Return 3 to 5 projects that would actually build understanding of this "
        "concept, ranging from beginner to advanced. For each project give:\n"
        "- scope: what to actually build (1-3 sentences)\n"
        "- skill_exercised: the specific skill/technique practiced\n"
        "- starter_pointer: a real, plausible starter repo or dataset (e.g. a "
        "well-known GitHub repo or HuggingFace dataset name)\n"
        "- difficulty: one of Beginner, Intermediate, Advanced\n\n"
        "Respond with ONLY strict JSON matching this exact schema, no markdown "
        "fences, no commentary:\n"
        '{"projects": [{"scope": "...", "skill_exercised": "...", '
        '"starter_pointer": "...", "difficulty": "..."}]}'
    )


def _extract_json(raw_result):
    """Claude's `result` text should be pure JSON, but strip markdown fences
    defensively in case a code block slipped in."""
    text = raw_result.strip()
    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    return json.loads(text)


def _projects_to_sections(projects):
    any_long = any(len(p.get("scope", "")) > _SCOPE_TABLE_LIMIT for p in projects)
    if not any_long:
        rows = [
            [
                p.get("scope", ""),
                p.get("skill_exercised", ""),
                p.get("starter_pointer", ""),
                p.get("difficulty", ""),
            ]
            for p in projects
        ]
        return [
            {
                "type": "table",
                "title": "Project ideas",
                "columns": _TABLE_COLUMNS,
                "rows": rows,
            }
        ]

    sections = []
    for i, p in enumerate(projects, start=1):
        body = (
            f"**Scope:** {p.get('scope', '')}\n\n"
            f"**Skill exercised:** {p.get('skill_exercised', '')}\n\n"
            f"**Starter pointer:** {p.get('starter_pointer', '')}\n\n"
            f"**Difficulty:** {p.get('difficulty', '')}"
        )
        sections.append({"type": "text", "title": f"Project {i}", "body": body})
    return sections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="{}")
    args = parser.parse_args()

    try:
        payload = json.loads(args.input)
    except json.JSONDecodeError as exc:
        envelope.emit(
            "projects", ok=False, summary="Invalid --input JSON",
            sections=[], error=f"Could not parse --input as JSON: {exc}",
        )
        return

    concept = (payload.get("concept") or "").strip()
    if not concept:
        envelope.progress("projects: no concept provided")
        envelope.emit(
            "projects",
            ok=False,
            summary="No concept provided",
            sections=[],
            error="Missing required input: 'concept' (e.g. \"transformer attention\").",
        )
        return

    envelope.progress(f"projects: asking Claude for project ideas on '{concept}'")
    try:
        response = claude.ask(_build_prompt(concept))
        parsed = _extract_json(response["result"])
        projects = parsed["projects"]
        if not isinstance(projects, list) or not projects:
            raise ValueError("Claude response contained no projects")
    except Exception as exc:  # noqa: BLE001 - degrade gracefully, never crash
        envelope.progress(f"projects: failed - {exc}")
        envelope.emit(
            "projects",
            ok=False,
            summary="Failed to generate project ideas",
            sections=[],
            error=str(exc),
        )
        return

    envelope.progress(f"projects: got {len(projects)} project ideas")
    sections = _projects_to_sections(projects)
    envelope.emit(
        "projects",
        ok=True,
        summary=f"{len(projects)} project ideas for '{concept}'",
        sections=sections,
    )


if __name__ == "__main__":
    main()
