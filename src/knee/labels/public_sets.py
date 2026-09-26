"""Public report-label sets: download, license check, load into one standard frame, leak check (S04 step 5).

    python -m knee.labels.public_sets download     # kaggle CLI → <cache_root>/labels/public/, writes manifest.json
    python -m knee.labels.public_sets check        # gold AUC per set + leak check (prints a table)

Standard frame (every reader, ours included): index = StudyInstanceUID, the 12 LABELS columns = P(finding) in [0,1],
**NaN = the reader says the report is silent on it** (never a hard 0). Readers without a silence signal (lixin) are
never NaN.

Which sets are readers (blended) and which are only references (docs/kaggle/labels_from_notebooks.md §1b):
- readers: pilkwang (Claude Opus 5, verdicts), steven v1 (one LLM, 0.5 = not addressed), lixin (GPT-"5.6-Sol",
  UNVERIFIED model), laymond (Qwen2.5-32B, *raw*, i.e. before its co-occurrence fills; blend.py applies our own).
- references (scored, never blended): steven v4 = steven v2 + lixin (would double-count lixin); JEV hybrid and JEV
  FINAL2 (per-finding choices/weights tuned on the gold 58, so their gold AUC is optimistic; FINAL2 is available as
  an ablation reader, `--readers ... jev_final2`).
- excluded, never downloaded: yunusgmsoy v5 and barun2104 (gold labels pasted into the 58 rows → AUC 1.0).
All five source datasets are CC0-1.0 (read with `kaggle datasets metadata` on 2026-09-26; re-checked on download).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from knee.paths import LABELS, load_paths


def _pilkwang(df: pd.DataFrame) -> pd.DataFrame:
    out = df.set_index("StudyInstanceUID")[LABELS].astype(float)
    silent = df.set_index("StudyInstanceUID")[[f"{k}__verdict" for k in LABELS]].eq("UNK").to_numpy()
    return out.mask(silent)


def _steven(df: pd.DataFrame) -> pd.DataFrame:
    out = df.set_index("StudyInstanceUID")[LABELS].astype(float)
    return out.mask(np.isclose(out.to_numpy(), 0.5))  # 0.5 = "report does not address it" (dataset card)


def _plain(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("StudyInstanceUID")[LABELS].astype(float)


def _laymond(df: pd.DataFrame) -> pd.DataFrame:
    d = df.set_index("StudyInstanceUID")
    out = d[[f"{k}__confidence" for k in LABELS]].astype(float)
    out.columns = LABELS
    silent = d[[f"{k}__mentioned" for k in LABELS]].eq(0).to_numpy()
    return out.mask(silent)


@dataclass(frozen=True)
class Source:
    name: str
    slug: str
    file: str
    parse: Callable[[pd.DataFrame], pd.DataFrame]
    role: str  # "reader" | "reference"
    note: str


SOURCES = [
    Source("pilkwang", "pilkwang/rsna-knee-llm-labels", "report_labels_v2.csv", _pilkwang, "reader",
           "Claude Opus 5 (low effort), YES/NO/UNK + severity; UNK → NaN"),
    Source("steven_v1", "stevenleehans/rsna-knee-llm-report-labels", "llm_labels_full.csv", _steven, "reader",
           "one LLM (model not stated); 0.5 → NaN"),
    Source("lixin", "lixin73/rsna-knee-llm-report-labels-sol56", "report_labels_gpt56sol.csv", _plain, "reader",
           "'GPT-5.6-Sol' (UNVERIFIED); no silence signal"),
    Source("laymond", "laymond/rsna-knee-abnormality-qwen3-8b-weak-labels", "weak_labels_v2_raw.csv", _laymond,
           "reader", "Qwen2.5-32B-Instruct 4-bit, before co-occurrence fills; mentioned=0 → NaN"),
    Source("steven_v4", "stevenleehans/rsna-knee-llm-report-labels", "llm_labels_v4_blend.csv", _plain, "reference",
           "steven v2 + lixin blend (double-counts lixin)"),
    Source("jev_hybrid", "riadmohamed42/jev-knee-labels", "HYBRID_labels_scores.csv", _plain, "reference",
           "per-finding winner of steven v4 / pilkwang chosen on the gold 58 (optimistic on gold)"),
    Source("jev_final2", "riadmohamed42/jev-knee-labels", "FINAL2_labels_scores.csv", _plain, "reference",
           "6-member OpenJev text ensemble; member weights from gold AUC (optimistic on gold)"),
]
BY_NAME = {s.name: s for s in SOURCES}
DEFAULT_READERS = [s.name for s in SOURCES if s.role == "reader"]


def public_dir(cache_root: Path) -> Path:
    return cache_root / "labels" / "public"


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def download(root: Path) -> dict:
    """Fetch each needed file with the Kaggle CLI (v2.x) and record license + file hash in manifest.json."""
    root.mkdir(parents=True, exist_ok=True)
    manifest = {"downloaded": time.strftime("%Y-%m-%d %H:%M"), "sources": {}}
    licenses: dict[str, list] = {}
    for s in SOURCES:
        d = root / s.slug.replace("/", "_")
        d.mkdir(exist_ok=True)
        if s.slug not in licenses:
            subprocess.run(["kaggle", "datasets", "metadata", s.slug, "-p", str(d)], check=True, capture_output=True)
            meta = json.loads((d / "dataset-metadata.json").read_text())
            meta = meta.get("info", meta)
            licenses[s.slug] = [x.get("name") for x in meta.get("licenses", [])]
        f = d / s.file
        if not f.exists():
            subprocess.run(["kaggle", "datasets", "download", s.slug, "-f", s.file, "-p", str(d), "-q"], check=True)
            z = d / (s.file + ".zip")
            if z.exists():  # the CLI zips large single files
                import zipfile
                with zipfile.ZipFile(z) as zf:
                    zf.extractall(d)
                z.unlink()
        manifest["sources"][s.name] = {"slug": s.slug, "file": s.file, "role": s.role, "note": s.note,
                                       "license": licenses[s.slug], "sha256": _sha256(f)}
        if licenses[s.slug] != ["CC0-1.0"]:
            print(f"WARNING {s.slug}: license {licenses[s.slug]} (expected CC0-1.0); check before using")
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def load(name: str, root: Path) -> pd.DataFrame:
    s = BY_NAME[name]
    df = pd.read_csv(root / s.slug.replace("/", "_") / s.file)
    out = s.parse(df).clip(0.0, 1.0)
    out = out[~out.index.duplicated()]
    out.index.name = "StudyInstanceUID"
    return out


def leak_check(frame: pd.DataFrame, gold: pd.DataFrame, auc_thr: float = 0.999, match_thr: float = 0.95) -> dict:
    """A set leaks when it reproduces the gold labels: near-perfect AUC on most findings, or binarized rows equal to
    the gold rows for almost every gold study. Report labels agree with gold at ~0.87–0.90 macro, never ~1.0."""
    from knee.utils.metrics import per_finding_auc
    g = gold.index.intersection(frame.index)
    y, p = gold.loc[g, LABELS].to_numpy(float), frame.loc[g, LABELS].to_numpy(float)
    aucs = per_finding_auc(y, np.nan_to_num(p, nan=0.5))
    rows_equal = float(np.mean(np.all((np.nan_to_num(p, nan=0.5) >= 0.5) == (y >= 0.5), axis=1))) if len(g) else 0.0
    n_perfect = int(np.sum(aucs >= auc_thr))
    return {"n_gold": len(g), "n_perfect_auc": n_perfect, "rows_equal": rows_equal,
            "leaked": bool(n_perfect >= 10 or rows_equal >= match_thr)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["download", "check"])
    ap.add_argument("--root", type=Path, help="default <cache_root>/labels/public")
    ap.add_argument("--train-csv", type=Path)
    a = ap.parse_args()
    if a.root is None or a.train_csv is None:
        p = load_paths()
        a.root = a.root or public_dir(p.cache_root)
        a.train_csv = a.train_csv or p.train_csv
    if a.cmd == "download":
        m = download(a.root)
        print(json.dumps({k: v["license"] for k, v in m["sources"].items()}, indent=1))
        return
    from knee.utils.metrics import bootstrap_macro_auc
    tr = pd.read_csv(a.train_csv)
    gold = tr[tr[LABELS].notna().all(axis=1)].set_index("StudyInstanceUID")[LABELS].astype(float)
    for s in SOURCES:
        f = load(s.name, a.root)
        g = gold.index.intersection(f.index)
        bs = bootstrap_macro_auc(gold.loc[g].to_numpy(), f.loc[g].fillna(0.5).to_numpy(), n_boot=500)
        lk = leak_check(f, gold)
        print(f"{s.name:11s} {s.role:9s} rows={len(f):5d} silent={f.isna().mean().mean():.3f} "
              f"gold={bs['auc']:.3f} [{bs['lo']:.3f},{bs['hi']:.3f}] leak={lk}")


if __name__ == "__main__":
    main()
