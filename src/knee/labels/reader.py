"""Our report reader: Claude API over the Message Batches endpoint (50% price), resumable, with a cost ledger.

    python -m knee.labels.reader sync   --n 3              # smoke test: 3 gold reports through the normal API
    python -m knee.labels.reader submit --split pilot      # 58 gold + 42 random (language-stratified) → one batch
        (pilot two models: add --model claude-sonnet-5 --run-name claude_sonnet5_v1 to every command of the 2nd)
    python -m knee.labels.reader collect                   # poll; write finished results to raw/ (re-run until done)
    python -m knee.labels.reader cost                      # $ so far + projection to all 4,407 reports
    python -m knee.labels.reader submit --split full       # only studies without a good raw result are sent
    python -m knee.labels.reader parse                     # raw/ → claude_<run>.parquet (one row per study×finding)

Auth: the SDK reads ANTHROPIC_API_KEY (put it in `.env` at the repo root, gitignored; it's loaded here) or an
`ant auth login` profile. The key is never printed or written anywhere.

Storage: `<cache_root>/labels/claude/<run_name>/` (MIRA-derived: the raw responses quote report text; never commit).
    ledger.jsonl        one line per submitted batch (id, model, custom_ids, collected flag)
    raw/<custom_id>.json one file per study: uid, model, prompt version, stop reason, usage, text or error
A study is "done" when its raw file has a parseable result for the current PROMPT_VERSION; everything else is
resubmitted by the next `submit`. custom_id = "r" + sha1(uid)[:24] (the API allows [A-Za-z0-9_-]{1,64}).

Budget [Dom, 2026-09-24]: ≤ ~$50 for all labeling. `submit --split full` refuses to start when the projection from
the pilot exceeds `--budget` (default 50) minus what was already spent.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from knee.labels import prompt as P
from knee.paths import LABELS, REPO_ROOT, load_paths

# $ per 1M tokens (input, output), standard API, from the claude-api skill's model table (cached 2026-06-24).
# UNVERIFIED against the live pricing page; the pilot's measured usage × these is what `cost` reports.
PRICES = {
    "claude-opus-5": (5.0, 25.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}
BATCH_DISCOUNT = 0.5
CACHE_WRITE_MULT = {"5m": 1.25, "1h": 2.0}
CACHE_READ_MULT = 0.1


def load_dotenv(path: Path = REPO_ROOT / ".env") -> None:
    """Minimal .env reader (KEY=VALUE lines) so the key never has to be typed into a shell history."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def custom_id(uid: str) -> str:
    return "r" + hashlib.sha1(uid.encode()).hexdigest()[:24]


def request_params(report: str, model: str, effort: str, thinking: str, max_tokens: int, ttl: str) -> dict:
    """One Messages API request body. The system prompt is identical for every study → prompt-cached."""
    params = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": P.SYSTEM_PROMPT, "cache_control": {"type": "ephemeral", "ttl": ttl}}],
        "messages": [{"role": "user", "content": P.user_message(report)}],
        "output_config": {"format": {"type": "json_schema", "schema": P.output_schema()}},
    }
    if effort:
        params["output_config"]["effort"] = effort
    if thinking == "adaptive":
        params["thinking"] = {"type": "adaptive"}
    elif thinking == "disabled":
        params["thinking"] = {"type": "disabled"}
    return params


# ---------------------------------------------------------------------------------------------------- run folder
class Run:
    def __init__(self, root: Path):
        self.root = root
        self.raw = root / "raw"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.ledger = root / "ledger.jsonl"

    def batches(self) -> list[dict]:
        if not self.ledger.exists():
            return []
        return [json.loads(x) for x in self.ledger.read_text(encoding="utf-8").splitlines() if x.strip()]

    def write_batches(self, rows: list[dict]) -> None:
        tmp = self.ledger.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        tmp.replace(self.ledger)

    def save_raw(self, cid: str, rec: dict) -> None:
        tmp = self.raw / f"{cid}.tmp"
        tmp.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.raw / f"{cid}.json")

    def load_raw(self) -> list[dict]:
        return [json.loads(f.read_text(encoding="utf-8")) for f in sorted(self.raw.glob("r*.json"))]

    def done_uids(self) -> set[str]:
        out = set()
        for r in self.load_raw():
            if r.get("prompt_version") == P.PROMPT_VERSION and r.get("text"):
                try:
                    P.parse_output(r["text"])
                    out.add(r["uid"])
                except (ValueError, KeyError, json.JSONDecodeError):
                    pass
        return out


def _usage_dict(u) -> dict:
    keys = ["input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens"]
    return {k: int(getattr(u, k, 0) or 0) for k in keys}


def _record(uid: str, model: str, mode: str, msg=None, error: str | None = None) -> dict:
    rec = {"uid": uid, "model": model, "mode": mode, "prompt_version": P.PROMPT_VERSION, "time": time.time(),
           "text": None, "error": error, "stop_reason": None, "usage": None, "served_by": None}
    if msg is not None:
        rec["stop_reason"] = msg.stop_reason
        rec["usage"] = _usage_dict(msg.usage)
        rec["served_by"] = getattr(msg, "model", None)
        if msg.stop_reason == "refusal":
            rec["error"] = "refusal"
        else:
            rec["text"] = next((b.text for b in msg.content if b.type == "text"), None)
            if msg.stop_reason == "max_tokens":
                rec["error"] = "max_tokens"
    return rec


# ---------------------------------------------------------------------------------------------------- selection
def load_reports(train_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(train_csv)
    df["is_gold"] = df[LABELS].notna().all(axis=1)
    return df[["StudyInstanceUID", "Report", "is_gold"]]


def pilot_uids(reports: pd.DataFrame, n_random: int = 42, seed: int = 0) -> list[str]:
    """All gold studies + n_random non-gold, spread over languages in proportion (so cost/agreement per language shows)."""
    from knee.audit.csv_audit import guess_language
    gold = reports[reports.is_gold].StudyInstanceUID.tolist()
    rest = reports[~reports.is_gold].copy()
    rest["lang"] = rest.Report.map(guess_language)
    rng = np.random.default_rng(seed)
    share = rest.lang.value_counts(normalize=True)
    picks: list[str] = []
    for lang, frac in share.items():
        k = max(1, int(round(frac * n_random)))
        pool = rest[rest.lang == lang].StudyInstanceUID.to_numpy()
        picks += list(rng.choice(pool, size=min(k, len(pool)), replace=False))
    return gold + picks[:n_random]


# ---------------------------------------------------------------------------------------------------- cost
def cost_of(rec: dict, ttl: str = "1h") -> float:
    u, model = rec.get("usage"), rec.get("model")
    if not u or model not in PRICES:
        return 0.0
    pin, pout = PRICES[model]
    usd = (u["input_tokens"] * pin + u["cache_creation_input_tokens"] * pin * CACHE_WRITE_MULT[ttl]
           + u["cache_read_input_tokens"] * pin * CACHE_READ_MULT + u["output_tokens"] * pout) / 1e6
    return usd * (BATCH_DISCOUNT if rec.get("mode") == "batch" else 1.0)


def cost_summary(run: Run, n_total: int) -> dict:
    recs = [r for r in run.load_raw() if r.get("usage")]
    spent = float(sum(cost_of(r) for r in recs))
    batch = [r for r in recs if r["mode"] == "batch"]
    per = float(np.mean([cost_of(r) for r in batch])) if batch else float("nan")
    done = len(run.done_uids())
    tok = pd.DataFrame([r["usage"] for r in recs]) if recs else pd.DataFrame()
    return {
        "responses": len(recs), "done_studies": done, "spent_usd": round(spent, 4),
        "usd_per_report_batch": round(per, 5) if np.isfinite(per) else None,
        "projected_remaining_usd": round(per * max(n_total - done, 0), 2) if np.isfinite(per) else None,
        "mean_tokens": {k: round(float(v), 1) for k, v in tok.mean().items()} if len(tok) else {},
        "errors": int(sum(1 for r in run.load_raw() if r.get("error"))),
    }


def total_spent(claude_dir: Path) -> float:
    """$ across every run folder (pilots of several models + full runs): the budget is for all labeling."""
    return float(sum(cost_of(json.loads(f.read_text(encoding="utf-8"))) for f in claude_dir.glob("*/raw/r*.json")))


# ---------------------------------------------------------------------------------------------------- commands
def cmd_sync(a, run: Run, reports: pd.DataFrame) -> None:
    import anthropic
    client = anthropic.Anthropic()
    todo = reports[reports.is_gold & ~reports.StudyInstanceUID.isin(run.done_uids())].head(a.n)
    for uid, rep in zip(todo.StudyInstanceUID, todo.Report):
        params = request_params(rep, a.model, a.effort, a.thinking, a.max_tokens, "5m")
        try:
            msg = client.messages.create(**params)
            rec = _record(uid, a.model, "sync", msg)
        except anthropic.APIStatusError as e:
            rec = _record(uid, a.model, "sync", error=f"{type(e).__name__} {e.status_code}")
        run.save_raw(custom_id(uid), rec)
        print(uid[-12:], rec["stop_reason"], rec["error"] or "ok", rec["usage"])


def cmd_submit(a, run: Run, reports: pd.DataFrame) -> None:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    uids = pilot_uids(reports, a.n_random, a.seed) if a.split == "pilot" else reports.StudyInstanceUID.tolist()
    pending = {cid for b in run.batches() if not b.get("collected") for cid in b["custom_ids"]}
    done = run.done_uids()
    todo = [u for u in uids if u not in done and custom_id(u) not in pending]
    if a.limit:
        todo = todo[: a.limit]
    if not todo:
        print("nothing to submit (all done or pending)")
        return
    if a.split == "full":
        cs = cost_summary(run, len(reports))
        per = cs["usd_per_report_batch"]
        if per is None:
            raise SystemExit("no batch pilot results yet: run the pilot (submit --split pilot, collect) first")
        proj = per * len(todo)
        spent = total_spent(run.root.parent)
        print(f"projection: {len(todo)} reports × ${per:.4f} = ${proj:.2f}; spent so far (all runs) ${spent:.2f}")
        if spent + proj > a.budget:
            raise SystemExit(f"over budget (${a.budget}); pass --budget to override after Dom's OK")
    text = dict(zip(reports.StudyInstanceUID, reports.Report))
    client = anthropic.Anthropic()
    for i in range(0, len(todo), a.chunk):
        part = todo[i: i + a.chunk]
        reqs = [Request(custom_id=custom_id(u), params=MessageCreateParamsNonStreaming(
            **request_params(text[u], a.model, a.effort, a.thinking, a.max_tokens, "1h"))) for u in part]
        batch = client.messages.batches.create(requests=reqs)
        rows = run.batches() + [{"batch_id": batch.id, "model": a.model, "split": a.split, "created": time.time(),
                                 "prompt_version": P.PROMPT_VERSION, "custom_ids": [custom_id(u) for u in part],
                                 "uids": part, "collected": False}]
        run.write_batches(rows)
        print(f"submitted batch {batch.id}: {len(part)} requests")


def cmd_collect(a, run: Run, reports: pd.DataFrame) -> None:
    import anthropic
    client = anthropic.Anthropic()
    rows = run.batches()
    for b in rows:
        if b.get("collected"):
            continue
        st = client.messages.batches.retrieve(b["batch_id"])
        print(f"{b['batch_id']}: {st.processing_status} {st.request_counts}")
        if st.processing_status != "ended":
            continue
        uid_of = dict(zip(b["custom_ids"], b["uids"]))
        for res in client.messages.batches.results(b["batch_id"]):
            uid = uid_of[res.custom_id]
            if res.result.type == "succeeded":
                rec = _record(uid, b["model"], "batch", res.result.message)
            else:
                err = res.result.type
                if res.result.type == "errored":
                    err = f"errored:{res.result.error.type}"
                rec = _record(uid, b["model"], "batch", error=err)
            run.save_raw(res.custom_id, rec)
        b["collected"] = True
        run.write_batches(rows)
    print(json.dumps(cost_summary(run, len(reports)), indent=1))


def parse_run(run: Run) -> pd.DataFrame:
    """raw/ → long frame: StudyInstanceUID, label, state, severity, degenerative, postoperative, bin, quote (+ study
    level language, knee_side). Latest good record per study wins."""
    rows = []
    best: dict[str, dict] = {}
    for r in run.load_raw():
        if r.get("prompt_version") == P.PROMPT_VERSION and r.get("text"):
            if r["uid"] not in best or r["time"] > best[r["uid"]]["time"]:
                best[r["uid"]] = r
    inv = {v: k for k, v in P.KEYS.items()}
    for uid, r in best.items():
        try:
            d = P.parse_output(r["text"])
        except (ValueError, KeyError, json.JSONDecodeError):
            continue
        for key, v in d["findings"].items():
            rows.append({"StudyInstanceUID": uid, "label": inv[key], "state": v["state"],
                         "severity": int(v["severity"]), "degenerative": bool(v["degenerative"]),
                         "postoperative": bool(v["postoperative"]), "quote": v["quote"],
                         "bin": P.to_bin(v["state"], v["severity"], v["degenerative"]),
                         "language": d["language"], "knee_side": d["knee_side"]})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sync", "submit", "collect", "cost", "parse"])
    ap.add_argument("--run-name", default="claude_v1")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--effort", default="low", help="low|medium|high|'' (the pilot decides; low keeps cost down)")
    ap.add_argument("--thinking", default="adaptive", choices=["adaptive", "disabled", "default"])
    ap.add_argument("--max-tokens", type=int, default=4000)
    ap.add_argument("--split", default="pilot", choices=["pilot", "full"])
    ap.add_argument("--n", type=int, default=3, help="sync: number of gold reports")
    ap.add_argument("--n-random", type=int, default=42)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=5000)
    ap.add_argument("--budget", type=float, default=50.0)
    ap.add_argument("--train-csv", type=Path)
    ap.add_argument("--out", type=Path, help="default <cache_root>/labels/claude")
    a = ap.parse_args()
    load_dotenv()
    if a.train_csv is None or a.out is None:
        p = load_paths()
        a.train_csv = a.train_csv or p.train_csv
        a.out = a.out or p.cache_root / "labels" / "claude"
    run = Run(a.out / a.run_name)
    reports = load_reports(a.train_csv)
    if a.cmd == "sync":
        cmd_sync(a, run, reports)
    elif a.cmd == "submit":
        cmd_submit(a, run, reports)
    elif a.cmd == "collect":
        cmd_collect(a, run, reports)
    elif a.cmd == "cost":
        print(json.dumps(cost_summary(run, len(reports)), indent=1))
    elif a.cmd == "parse":
        df = parse_run(run)
        out = a.out / f"{a.run_name}.parquet"
        df.to_parquet(out, index=False)
        print(f"{df.StudyInstanceUID.nunique()} studies → {out}")


if __name__ == "__main__":
    main()
