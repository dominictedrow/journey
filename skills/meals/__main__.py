"""Macro-filtered meal ideas + config.json.

Invocation:
    python -m skills.meals --input '{"craving":"chicken"}'
    python -m skills.meals --input '{}'

`craving` is optional plain text (may be empty/absent).

Hard constraints (protein floor, calorie band, required vitamins, avoid-list)
live in config.json and are enforced twice: once in the prompt sent to Claude,
and again in Python against every meal Claude returns. Per plan.md, "Claude's
arithmetic is not trusted" — any meal that fails re-validation is dropped from
the rendered table and counted in a separate rejected-count, rather than
rendered as if it were verified.
"""

import argparse
import json
from pathlib import Path

from .._lib import claude, envelope

CONFIG_PATH = Path(__file__).parent / "config.json"


def load_config():
    with open(CONFIG_PATH) as f:
        return json.load(f)


def build_prompt(config, craving):
    craving_line = (
        f'Dom is craving: "{craving}".' if craving else "Dom has no specific craving right now."
    )
    avoid = config.get("avoid") or []
    avoid_line = f"Avoid these ingredients/allergens entirely: {', '.join(avoid)}." if avoid else ""

    return f"""Suggest 3 to 5 meal ideas for Dom that satisfy ALL of these hard constraints:
- Protein: at least {config['protein_min_g']}g per meal
- Calories: between {config['calorie_min']} and {config['calorie_max']} per meal
- Must plausibly cover at least one of these required vitamins: {', '.join(config['required_vitamins'])}
{avoid_line}
{craving_line}

Respond with ONLY strict JSON (no markdown fences, no commentary) matching exactly this schema:
{{
  "meals": [
    {{
      "name": "string",
      "protein_g": number,
      "calories": number,
      "vitamins_covered": ["string", ...],
      "recipe": "short string, a few sentences"
    }}
  ]
}}"""


def _parse_claude_json(raw_result):
    """Claude's `result` text should be strict JSON; strip stray markdown fences defensively."""
    text = raw_result.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)


def validate_meal(meal, config):
    """Re-check one meal dict against config's hard constraints.

    Returns (is_valid, reason). Never trusts Claude's stated numbers as fact
    beyond checking they satisfy the constraints themselves.
    """
    try:
        protein_g = float(meal["protein_g"])
        calories = float(meal["calories"])
    except (KeyError, TypeError, ValueError):
        return False, "missing/non-numeric protein_g or calories"

    if protein_g < config["protein_min_g"]:
        return False, f"protein {protein_g}g < floor {config['protein_min_g']}g"

    if not (config["calorie_min"] <= calories <= config["calorie_max"]):
        return False, f"calories {calories} outside band {config['calorie_min']}-{config['calorie_max']}"

    vitamins_covered = meal.get("vitamins_covered") or []
    required = set(config["required_vitamins"])
    if required and not (set(vitamins_covered) & required):
        return False, f"no overlap between vitamins_covered={vitamins_covered} and required={sorted(required)}"

    avoid = set(config.get("avoid") or [])
    if avoid:
        name_and_recipe = f"{meal.get('name', '')} {meal.get('recipe', '')}".lower()
        hit = next((a for a in avoid if a.lower() in name_and_recipe), None)
        if hit:
            return False, f"mentions avoided item {hit!r}"

    return True, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="{}")
    args = parser.parse_args()

    payload = json.loads(args.input)
    craving = (payload.get("craving") or "").strip()

    config = load_config()

    envelope.progress("meals: loaded config", config)
    envelope.progress(f"meals: asking Claude (craving={craving!r})")

    prompt = build_prompt(config, craving)
    response = claude.ask(prompt)

    try:
        parsed = _parse_claude_json(response["result"])
        candidate_meals = parsed.get("meals", [])
    except (KeyError, json.JSONDecodeError) as e:
        envelope.progress(f"meals: failed to parse Claude response: {e}")
        envelope.emit(
            "meals",
            ok=False,
            summary="Claude response could not be parsed as JSON",
            sections=[],
            error=str(e),
        )
        return

    envelope.progress(f"meals: got {len(candidate_meals)} candidate meal(s), validating")

    accepted, rejected = [], []
    for meal in candidate_meals:
        is_valid, reason = validate_meal(meal, config)
        if is_valid:
            accepted.append(meal)
        else:
            rejected.append((meal, reason))
            envelope.progress(f"meals: rejected {meal.get('name', '<unnamed>')!r}: {reason}")

    sections = []

    if accepted:
        sections.append(
            {
                "type": "table",
                "title": "Meal ideas",
                "columns": ["Meal", "Protein (g)", "Calories", "Vitamins"],
                "rows": [
                    [
                        m.get("name", ""),
                        m.get("protein_g", ""),
                        m.get("calories", ""),
                        ", ".join(m.get("vitamins_covered", [])),
                    ]
                    for m in accepted
                ],
            }
        )
        sections.append(
            {
                "type": "text",
                "title": "Recipes",
                "body": "\n\n".join(
                    f"**{m.get('name', 'Untitled')}**\n\n{m.get('recipe', '')}" for m in accepted
                ),
            }
        )

    if rejected:
        sections.append(
            {
                "type": "text",
                "title": f"Rejected — failed macro validation ({len(rejected)})",
                "body": "\n".join(
                    f"- **{m.get('name', '<unnamed>')}**: {reason}" for m, reason in rejected
                ),
            }
        )

    summary = f"{len(accepted)} meal(s) validated"
    if rejected:
        summary += f", {len(rejected)} rejected for failing macro checks"

    envelope.progress("meals: done")
    envelope.emit(
        "meals",
        ok=True,
        summary=summary,
        sections=sections,
    )


if __name__ == "__main__":
    main()
