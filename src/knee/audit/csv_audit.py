"""S01 step B1–B3: counts, gold-label prevalence, series metadata crosstabs, report statistics.

Reads only the CSVs (small). Writes <audit_out>/csv_audit.json and <audit_out>/report_groups.csv
(study → normalized-report hash, used for GroupKFold in S04).

    python -m knee.audit.csv_audit            # paths from configs/paths.yaml
    python -m knee.audit.csv_audit --data-root D:/rsna --out C:/knee_cache/audit
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from knee.paths import LABELS, load_paths

# Rough language guess: script first, then stop-word votes for Latin-script languages.
# Good enough for a distribution table; not used for anything downstream.
_STOP = {
    "en": {"the", "of", "and", "is", "with", "no", "there", "are", "in", "normal", "findings", "impression"},
    "es": {"de", "la", "el", "y", "en", "con", "sin", "los", "las", "del", "se", "rodilla"},
    "tr": {"ve", "ile", "bir", "olarak", "izlendi", "mevcuttur", "sinyal", "diz", "değişiklik", "yok"},
    "de": {"der", "die", "und", "das", "mit", "kein", "keine", "ist", "im", "des", "knie", "sich"},
    "nl": {"de", "het", "van", "een", "en", "geen", "met", "is", "op", "knie", "niet"},
    "fr": {"de", "la", "le", "et", "des", "du", "les", "sans", "avec", "genou", "une"},
    "it": {"di", "il", "la", "e", "con", "del", "della", "non", "ginocchio", "si"},
    "pt": {"de", "do", "da", "e", "com", "sem", "os", "joelho", "não", "em"},
    "hr/sr/bs": {"i", "je", "se", "u", "na", "bez", "koljena", "od", "su", "nije", "koljeno", "zglobu"},
}


def guess_language(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "empty"
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "unknown"
    scripts = Counter()
    for c in letters:
        name = unicodedata.name(c, "")
        scripts["greek" if name.startswith("GREEK") else "cyrillic" if name.startswith("CYRILLIC") else "latin"] += 1
    script, n = scripts.most_common(1)[0]
    if script != "latin" and n / len(letters) > 0.5:
        return script  # 'greek' or 'cyrillic' (Bulgarian/Russian/Serbian not separated)
    words = re.findall(r"\w+", text.lower())
    votes = {lang: sum(w in sw for w in words) for lang, sw in _STOP.items()}
    best = max(votes, key=votes.get)
    return best if votes[best] >= 3 else "unknown"


def normalize_report(text: str) -> str:
    """Case/punctuation/whitespace-insensitive form. Raw MD5 splits 25 duplicate groups across folds; this doesn't ([101])."""
    t = unicodedata.normalize("NFKC", str(text)).lower()
    t = re.sub(r"[^\w]+", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def report_hash(text: str) -> str:
    return hashlib.md5(normalize_report(text).encode("utf-8")).hexdigest()[:16]


def _q(s: pd.Series) -> dict:
    return {k: float(v) for k, v in s.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).items()}


def audit(data_root: Path, out_dir: Path) -> dict:
    train = pd.read_csv(data_root / "train.csv")
    series = pd.read_csv(data_root / "train_series.csv")
    res: dict = {}

    # B1 counts
    lab = train[LABELS]
    n_labeled = lab.notna().sum(axis=1)
    res["train_csv"] = {
        "rows": int(len(train)), "columns": list(train.columns),
        "unique_studies": int(train.StudyInstanceUID.nunique()),
        "studies_all_12_labels": int((n_labeled == 12).sum()),
        "studies_no_labels": int((n_labeled == 0).sum()),
        "studies_partial_labels": int(((n_labeled > 0) & (n_labeled < 12)).sum()),
        "empty_reports": int(train.Report.fillna("").str.strip().eq("").sum()),
    }
    per_study = series.groupby("StudyInstanceUID").size()
    res["train_series_csv"] = {
        "rows": int(len(series)), "unique_series": int(series.SeriesInstanceUID.nunique()),
        "studies": int(series.StudyInstanceUID.nunique()),
        "studies_in_train_csv_without_series": int(len(set(train.StudyInstanceUID) - set(series.StudyInstanceUID))),
        "studies_in_series_csv_not_in_train_csv": int(len(set(series.StudyInstanceUID) - set(train.StudyInstanceUID))),
        "series_per_study": _q(per_study),
    }
    for name in ("test.csv", "test_series.csv", "sample_submission.csv"):
        p = data_root / name
        if p.exists():
            df = pd.read_csv(p)
            res[name] = {"rows": int(len(df)), "columns": list(df.columns)}

    # B2 gold prevalence
    gold = train[n_labeled == 12]
    res["gold"] = {
        "n": int(len(gold)),
        "positives": {c: int(gold[c].sum()) for c in LABELS},
        "values_seen": sorted({float(v) for v in np.unique(gold[LABELS].to_numpy())}),
        "findings_per_study": _q(gold[LABELS].sum(axis=1)),
        "all_negative_studies": int((gold[LABELS].sum(axis=1) == 0).sum()),
    }

    # B3 series metadata
    s = series.copy()
    res["series_meta"] = {
        "fluid_equals_fatsup_rows": int((s.Fluid_Sensitive == s.Fat_Suppression).sum()),
        "rows": int(len(s)),
        "plane_counts": {k: int(v) for k, v in s.Anatomical_Plane.value_counts().items()},
        "crosstab_plane_fluid_fatsup": {
            f"{p}|fluid={f}|fatsup={fs}": int(n)
            for (p, f, fs), n in s.groupby(["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]).size().items()
        },
    }
    planes = s.groupby("StudyInstanceUID").Anatomical_Plane.apply(set)
    res["series_meta"]["studies_missing_plane"] = {
        p: int((~planes.apply(lambda x, p=p: p in x)).sum()) for p in ("Sagittal", "Coronal", "Axial")
    }
    layout = s.groupby("StudyInstanceUID").Anatomical_Plane.apply(
        lambda x: f"{(x == 'Sagittal').sum()}/{(x == 'Coronal').sum()}/{(x == 'Axial').sum()}")
    res["series_meta"]["top_sag_cor_ax_layouts"] = {k: int(v) for k, v in layout.value_counts().head(10).items()}
    slot = s.assign(k=s.Anatomical_Plane.str[:3] + "_" + np.where(s.Fluid_Sensitive == 1, "fluid", "other"))
    has = slot.groupby("StudyInstanceUID").k.apply(set)
    res["series_meta"]["study_has_slot_pct"] = {
        k: round(100 * has.apply(lambda x, k=k: k in x).mean(), 2) for k in sorted(slot.k.unique())
    }

    # Reports: length, language, duplicates
    rep = train.Report.fillna("")
    res["reports"] = {"chars": _q(rep.str.len()), "words": _q(rep.str.split().str.len())}
    langs = rep.apply(guess_language)
    res["reports"]["language_guess"] = {k: int(v) for k, v in langs.value_counts().items()}
    res["reports"]["non_ascii_pct"] = round(100 * rep.apply(lambda t: any(ord(c) > 127 for c in t)).mean(), 2)
    h = rep.apply(report_hash)
    raw = rep.apply(lambda t: hashlib.md5(t.encode("utf-8")).hexdigest())
    grp = h.map(h.value_counts())
    res["reports"]["duplicates"] = {
        "distinct_raw": int(raw.nunique()), "distinct_normalized": int(h.nunique()),
        "studies_sharing_text": int((grp > 1).sum()), "groups_with_2plus": int((h.value_counts() > 1).sum()),
        "largest_group": int(grp.max()),
        "gold_studies_sharing_text": int(((grp > 1) & (n_labeled == 12)).sum()),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"StudyInstanceUID": train.StudyInstanceUID, "report_hash": h, "language_guess": langs,
                  "is_gold": n_labeled == 12}).to_csv(out_dir / "report_groups.csv", index=False)
    (out_dir / "csv_audit.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.data_root is None or a.out is None:
        p = load_paths()
        a.data_root = a.data_root or p.data_root
        a.out = a.out or p.audit_out
    res = audit(a.data_root, a.out)
    print(json.dumps({k: res[k] for k in ("train_csv", "gold")}, indent=2))


if __name__ == "__main__":
    main()
