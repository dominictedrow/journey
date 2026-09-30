"""Our report reader without the API: Claude Code agents read the reports, this module does everything else.

Same instructions (prompt.SYSTEM_PROMPT), same output schema and the same raw/ record layout as reader.py, so
`reader parse` and `build_labels` work on an agent run unchanged. Decided 2026-09-27 (Dom has a subscription, no API
credits). The full process, the dispatch prompt and how to reproduce a run: docs/labels_agent_reader.md.

    python -m knee.labels.agent_reader export --split pilot    # 58 gold + 42 random → shards/sNNNN.jsonl (25 each)
    python -m knee.labels.agent_reader prompt s0000 s0001      # the exact text to give one Claude Code agent
    (agents write out/sNNNN.jsonl, one compact line per report)
    python -m knee.labels.agent_reader ingest                  # validate out/ → raw/<custom_id>.json
    python -m knee.labels.agent_reader status                  # shards without output, studies done, rejects
    python -m knee.labels.agent_reader export --split scan --run-name scan_v2   # 100 non-gold, non-pilot reports
    python -m knee.labels.agent_reader prompt --scan s0000 --run-name scan_v2   # reply = notes on rule gaps
    python -m knee.labels.reader parse --run-name agent_v1     # then build_labels --claude-run agent_v1

Blinding: a shard holds only the custom_id (sha1 of the uid) and the report text, shuffled, so an agent can't tell
gold studies from the rest; the dispatch prompt forbids opening any other file (train.csv carries the gold labels).

Storage: `<cache_root>/labels/claude/<run_name>/` (MIRA-derived; never commit)
    AGENT_INSTRUCTIONS.md   rendered from this module + prompt.py (the agent reads it first)
    shards/sNNNN.jsonl      {"id": custom_id, "report": text} per line
    out/sNNNN.jsonl         the agent's answers (compact format, see AGENT_FORMAT)
    raw/<custom_id>.json    one validated record per study, reader.py layout, mode="agent"
    ingest_log.json         per shard: accepted / rejected / missing ids from the last ingest
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from knee.labels import prompt as P
from knee.audit.csv_audit import guess_language
from knee.labels.reader import Run, custom_id, load_reports, pilot_uids
from knee.paths import load_paths

AGENT_FORMAT_VERSION = "c1"
FIELDS = ["state", "severity", "degenerative", "postoperative", "quote"]

AGENT_FORMAT = """## Output format (compact; one JSON object per line, one line per report, same order as the shard)

{"id": "<id from the shard>", "language": "<English name>", "knee_side": "left|right|unknown",
 "f": {"<finding key>": [state, severity, degenerative, postoperative, "quote"], ...}}

- Finding keys: acl, mcl, medial_meniscus, lateral_meniscus, medial_oa, lateral_oa, pf_oa, effusion, synovitis,
  bakers_cyst, contusion, fracture.
- A finding you leave out of "f" means state "not_mentioned" (severity 0, false, false, ""). Only leave a finding
  out when the report really never addresses it or its structure; "normal"/"intact" is "absent" and must be listed.
- state is one of "present", "absent", "uncertain", "not_mentioned"; severity an integer 0-3; the two flags are
  true/false; quote a verbatim phrase of at most 12 words in the report's language (JSON-escape quotes).
- Each line must be valid JSON on a single physical line. No commentary, no markdown fences, nothing else in the file.

Example line (made up):
{"id": "r0123abcd", "language": "Spanish", "knee_side": "right", "f": {"acl": ["absent", 0, false, false, "LCA de morfología y señal normal"], "medial_meniscus": ["present", 3, false, false, "rotura compleja del cuerno posterior"], "effusion": ["present", 1, false, false, "leve derrame articular"]}}
"""

AGENT_RULES = """## How to work (these rules are part of the method; follow them exactly)

1. Read every report yourself, one at a time, and decide each finding by reading it. Do NOT write or run code,
   regexes or keyword lists to label reports, and do not copy another report's answer: the method is "an LLM reads
   each report with these instructions", and the write-up claims exactly that.
2. Open only this file and the shard files you were given. Do not open train.csv or any other file in the project:
   some of them contain labels for these studies, and seeing them would contaminate the validation.
3. Write your answers for shard sNNNN to out/sNNNN.jsonl (same folder level as shards/), one line per report in the
   shard, in shard order, every id exactly once. Write each shard's file only when that shard is complete.
4. When in doubt, follow the conventions in the instructions above literally; do not invent your own.
"""


def instructions() -> str:
    return ("# Knee MRI report reader: instructions for a Claude Code agent\n\n"
            f"prompt {P.PROMPT_VERSION}, agent format {AGENT_FORMAT_VERSION}\n\n"
            "## Task (identical to the API reader's system prompt)\n\n" + P.SYSTEM_PROMPT + "\n\n"
            + AGENT_FORMAT + "\n" + AGENT_RULES)


def scan_uids(reports: pd.DataFrame, n: int = 100, seed: int = 1, pilot_n_random: int = 42) -> list[str]:
    """Non-gold studies outside the default pilot, language-stratified: a sample for finding prompt gaps.

    Prompt rules drawn from the agents' notes on these reports never saw the pilot's reports, so a re-pilot of the new
    prompt on the 58 gold is a cleaner test than v2's was (v2's rules came from notes on the pilot reports themselves).
    """
    pilot = set(pilot_uids(reports, pilot_n_random, 0))
    rest = reports[~reports.is_gold & ~reports.StudyInstanceUID.isin(pilot)].copy()
    rest["lang"] = rest.Report.map(guess_language)
    rng = np.random.default_rng(seed)
    picks: list[str] = []
    for lang, frac in rest.lang.value_counts(normalize=True).items():
        pool = rest[rest.lang == lang].StudyInstanceUID.to_numpy()
        picks += list(rng.choice(pool, size=min(max(1, int(round(frac * n))), len(pool)), replace=False))
    return picks[:n]


SCAN_NOTES = ("Also list the places where these instructions left you unsure how to label a report: a missing rule, "
              "two rules that conflict, or wording the scales don't cover. Give up to 10 short notes in total, each "
              "naming the finding, the kind of phrase (paraphrase it; no ids, no full quotes) and what you chose.")


def dispatch_prompt(run_dir: Path, shards: list[str], scan: bool = False) -> str:
    """The exact message one agent gets. Kept in code so a reproduction uses the same words.

    scan=True (a `--split scan` run) swaps the short reply for up to 10 notes on gaps in the rules."""
    files = "\n".join(f"- {run_dir / 'shards' / (s + '.jsonl')}  →  write {run_dir / 'out' / (s + '.jsonl')}"
                      for s in shards)
    return (f"You are labeling knee MRI radiology reports for a research dataset.\n\n"
            f"First read the instructions: {run_dir / 'AGENT_INSTRUCTIONS.md'}\n"
            f"Then label every report in these shard files (each line is {{\"id\", \"report\"}}):\n{files}\n\n"
            "Follow the instructions' rules exactly: read each report yourself (no code, regexes or keyword "
            "matching), open no other files, and write one compact JSON line per report to the output file named "
            "next to each shard. When done, reply with one line per shard: shard name, number of lines written, and "
            + (f"nothing else on that line. {SCAN_NOTES}" if scan else
               "anything you were unsure about (at most 3 short notes).")
            + " Do not paste the answers into your reply.")


# ------------------------------------------------------------------------------------------------ compact ↔ canonical
def expand(line: dict) -> dict:
    """Compact agent line → the canonical API response dict (prompt.output_schema). Raises ValueError on bad input."""
    f = line.get("f")
    if not isinstance(f, dict):
        raise ValueError("missing 'f'")
    unknown = set(f) - set(P.KEYS.values())
    if unknown:
        raise ValueError(f"unknown finding keys {sorted(unknown)}")
    side = line.get("knee_side", "unknown")
    if side not in ("left", "right", "unknown"):
        raise ValueError(f"bad knee_side {side!r}")
    findings = {}
    for k in P.KEYS.values():
        v = f.get(k, ["not_mentioned", 0, False, False, ""])
        if not isinstance(v, list) or len(v) != 5:
            raise ValueError(f"{k}: expected [state, severity, degenerative, postoperative, quote]")
        st, sev, deg, post, quote = v
        if st not in P.STATES:
            raise ValueError(f"{k}: bad state {st!r}")
        if isinstance(sev, bool) or not isinstance(sev, int) or not 0 <= sev <= 3:
            raise ValueError(f"{k}: bad severity {sev!r}")
        if not isinstance(deg, bool) or not isinstance(post, bool) or not isinstance(quote, str):
            raise ValueError(f"{k}: bad flag/quote types")
        findings[k] = dict(zip(FIELDS, v))
    return {"language": str(line.get("language", "")), "knee_side": side, "findings": findings}


def _record(uid: str, model: str, shard: str, canonical: dict) -> dict:
    """reader.py's raw record layout; usage None (subscription, no per-call token counts)."""
    return {"uid": uid, "model": model, "mode": "agent", "prompt_version": P.PROMPT_VERSION, "time": time.time(),
            "text": json.dumps(canonical, ensure_ascii=False), "error": None, "stop_reason": "agent", "usage": None,
            "served_by": "claude-code-agent", "agent_format": AGENT_FORMAT_VERSION, "shard": shard}


# ------------------------------------------------------------------------------------------------ commands
def cmd_export(a, run: Run, reports: pd.DataFrame) -> None:
    shard_dir, out_dir = run.root / "shards", run.root / "out"
    shard_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    (run.root / "AGENT_INSTRUCTIONS.md").write_text(instructions())
    existing = sorted(shard_dir.glob("s*.jsonl"))
    open_ids = {json.loads(x)["id"] for s in existing if not (out_dir / s.name).exists()
                for x in s.read_text().splitlines() if x.strip()}
    uids = (pilot_uids(reports, a.n_random, a.seed) if a.split == "pilot"
            else scan_uids(reports, a.n_scan, a.seed + 1, a.n_random) if a.split == "scan"
            else reports.StudyInstanceUID.tolist())
    done = run.done_uids()
    todo = [u for u in uids if u not in done and custom_id(u) not in open_ids]
    if not todo:
        print("nothing to export (all done or already in a shard without output)")
        return
    rng = np.random.default_rng(a.seed + len(existing))
    todo = [todo[i] for i in rng.permutation(len(todo))]  # blinding: gold studies are not first
    text = dict(zip(reports.StudyInstanceUID, reports.Report))
    start = int(existing[-1].stem[1:]) + 1 if existing else 0
    names = []
    for j, i in enumerate(range(0, len(todo), a.shard_size)):
        name = f"s{start + j:04d}"
        lines = [json.dumps({"id": custom_id(u), "report": text[u].strip()}, ensure_ascii=False)
                 for u in todo[i: i + a.shard_size]]
        (shard_dir / f"{name}.jsonl").write_text("\n".join(lines) + "\n")
        names.append(name)
    print(f"exported {len(todo)} reports → {len(names)} shards {names[0]}..{names[-1]} in {shard_dir}")


def cmd_ingest(a, run: Run, reports: pd.DataFrame) -> None:
    uid_of = {custom_id(u): u for u in reports.StudyInstanceUID}
    log = {}
    for out in sorted((run.root / "out").glob("s*.jsonl")):
        shard = run.root / "shards" / out.name
        want = [json.loads(x)["id"] for x in shard.read_text().splitlines() if x.strip()] if shard.exists() else []
        acc, rej, seen = 0, [], set()
        for n, raw in enumerate(out.read_text().splitlines(), 1):
            if not raw.strip():
                continue
            try:
                line = json.loads(raw)
                cid = line["id"]
                if cid not in want:
                    raise ValueError(f"id {cid!r} not in shard")
                canonical = expand(line)
                P.parse_output(json.dumps(canonical))
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as e:
                rej.append(f"line {n}: {e}")
                continue
            run.save_raw(cid, _record(uid_of[cid], a.model, out.stem, canonical))
            seen.add(cid)
            acc += 1
        log[out.stem] = {"accepted": acc, "rejected": rej, "missing": [c for c in want if c not in seen]}
    (run.root / "ingest_log.json").write_text(json.dumps(log, indent=1))
    bad = {k: v for k, v in log.items() if v["rejected"] or v["missing"]}
    print(f"ingested {sum(v['accepted'] for v in log.values())} lines from {len(log)} shard outputs; "
          f"{len(bad)} shards with rejects/missing (details in ingest_log.json)")
    for k, v in bad.items():
        print(f"  {k}: {len(v['rejected'])} rejected, {len(v['missing'])} missing; {v['rejected'][:2]}")


def cmd_status(a, run: Run, reports: pd.DataFrame) -> None:
    shards = sorted((run.root / "shards").glob("s*.jsonl"))
    no_out = [s.stem for s in shards if not (run.root / "out" / s.name).exists()]
    print(json.dumps({"shards": len(shards), "shards_without_output": no_out,
                      "studies_done": len(run.done_uids()), "studies_total": len(reports)}, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["export", "prompt", "ingest", "status"])
    ap.add_argument("shards", nargs="*", help="prompt: shard names (e.g. s0000 s0001)")
    ap.add_argument("--run-name", default="agent_v1")
    ap.add_argument("--model", default="claude-opus-5-5", help="the model the agents ran on (recorded per study)")
    ap.add_argument("--split", default="pilot", choices=["pilot", "full", "scan"],
                    help="scan: n-scan non-gold reports outside the pilot, for finding prompt gaps")
    ap.add_argument("--n-scan", type=int, default=100)
    ap.add_argument("--scan", action="store_true", help="prompt: the scan run's reply (notes on rule gaps)")
    ap.add_argument("--shard-size", type=int, default=25)
    ap.add_argument("--n-random", type=int, default=42)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--train-csv", type=Path)
    ap.add_argument("--out", type=Path, help="default <cache_root>/labels/claude")
    a = ap.parse_args()
    if a.train_csv is None or a.out is None:
        p = load_paths()
        a.train_csv = a.train_csv or p.train_csv
        a.out = a.out or p.cache_root / "labels" / "claude"
    run = Run(a.out / a.run_name)
    if a.cmd == "prompt":
        print(dispatch_prompt(run.root, a.shards, a.scan))
        return
    reports = load_reports(a.train_csv)
    {"export": cmd_export, "ingest": cmd_ingest, "status": cmd_status}[a.cmd](a, run, reports)


if __name__ == "__main__":
    main()
