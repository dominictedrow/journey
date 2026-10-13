"""Build the label + fold files every track trains on, and the label QA report (S04 steps 6–9).

    python -m knee.labels.build_labels                       # all public readers (+ our Claude reader if parsed)
    python -m knee.labels.build_labels --version v0 --no-claude   # public-only labels (usable before the API run)

Writes to `<cache_root>/labels/` (MIRA-derived, never committed):
    labels_<ver>.parquet   StudyInstanceUID, 12 soft targets, 12 w_<label>, is_gold   (format fixed by train_sup.py)
    folds_<ver>.csv        StudyInstanceUID, fold
    folds_<ver>_detail.csv + report_hash, group, stratum
    qa_<ver>.json          every number in the report
    disagreements_<ver>.md the skim list for Dom, with the reader's quotes (report text → stays out of the repo)
and `docs/labels.md` (numbers only, no report text; safe to commit).

Gold rows keep their **report-derived** labels here. The 58 are only ever scored (out of fold); whether to also
train on them (e.g. at 2–3× weight, cross-fitted) is a trainer decision (S05), not baked into this file.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from knee.labels import public_sets as PS
from knee.labels.blend import BlendConfig, blend, reader_for_scoring
from knee.labels.calibrate import reader_frame
from knee.labels.folds import check_folds, make_folds
from knee.paths import LABELS, REPO_ROOT, load_paths
from knee.utils.metrics import bootstrap_macro_auc, df_to_markdown, per_finding_auc

N_BOOT = 1000


def gold_frame(train: pd.DataFrame) -> pd.DataFrame:
    g = train[train[LABELS].notna().all(axis=1)]
    return g.set_index("StudyInstanceUID")[LABELS].astype(float)


def score(gold: pd.DataFrame, pred: pd.DataFrame) -> dict:
    g = gold.index
    return bootstrap_macro_auc(gold.to_numpy(), pred.reindex(g)[LABELS].to_numpy(), n_boot=N_BOOT)


def _fmt(bs: dict) -> str:
    return f"{bs['auc']:.3f} [{bs['lo']:.3f}, {bs['hi']:.3f}]"


def spearman_matrix(frames: dict[str, pd.DataFrame], uids: list[str]) -> pd.DataFrame:
    """Mean over findings of the Spearman correlation between two readers, on cells where both spoke."""
    names = list(frames)
    M = pd.DataFrame(np.nan, index=names, columns=names)
    for a in names:
        for b in names:
            rs = []
            for k in LABELS:
                x, y = frames[a].reindex(uids)[k], frames[b].reindex(uids)[k]
                ok = x.notna() & y.notna()
                if ok.sum() > 20 and x[ok].nunique() > 1 and y[ok].nunique() > 1:
                    rs.append(x[ok].rank().corr(y[ok].rank()))
            M.loc[a, b] = float(np.mean(rs)) if rs else np.nan
    return M


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", default="v1")
    ap.add_argument("--readers", nargs="*", default=PS.DEFAULT_READERS)
    ap.add_argument("--claude-run", default="claude_v1")
    ap.add_argument("--no-claude", action="store_true")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--weight-mode", default="agree_cert", choices=["agree_cert", "certainty", "ones"])
    ap.add_argument("--silent-cell-p", type=float, default=None, help="v1: 0.15 (E016); default = v0 behaviour")
    ap.add_argument("--site-mode", default="stratify", choices=["stratify", "group", "none"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--docs", type=Path, default=REPO_ROOT / "docs" / "labels.md")
    a = ap.parse_args()
    paths = load_paths()
    out = paths.cache_root / "labels"
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(paths.train_csv)
    uids = train.StudyInstanceUID.tolist()
    gold = gold_frame(train)
    nongold = [u for u in uids if u not in gold.index]
    root = PS.public_dir(paths.cache_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8")) if (root / "manifest.json").exists() else {}

    frames = {n: PS.load(n, root) for n in a.readers}
    refs = {s.name: PS.load(s.name, root) for s in PS.SOURCES if s.role == "reference" and s.name not in frames}
    leaks = {n: PS.leak_check(f, gold) for n, f in {**frames, **refs}.items()}
    bad = [n for n, lk in leaks.items() if lk["leaked"] and n in frames]
    if bad:
        raise SystemExit(f"leaked label sets in the reader list: {bad}")

    claude_long, cmaps, cost = None, {}, None
    cpath = paths.cache_root / "labels" / "claude" / f"{a.claude_run}.parquet"
    if not a.no_claude and cpath.exists():
        claude_long = pd.read_parquet(cpath)
        frames["claude"], cmaps = reader_frame(claude_long, gold, calibrate=not a.no_calibrate)
        try:
            from knee.labels.reader import Run, cost_summary
            cost = cost_summary(Run(paths.cache_root / "labels" / "claude" / a.claude_run), len(uids))
        except Exception as e:  # cost is informational only
            cost = {"error": str(e)}
    elif not a.no_claude:
        print(f"note: {cpath} not found → building without our Claude reader")

    cfg = BlendConfig(weight_mode=a.weight_mode, silent_cell_p=a.silent_cell_p)
    p, w, status = blend(frames, uids, cfg)

    # ---- gold ablations (all out-of-fold w.r.t. gold: nothing below is fitted on gold except the out-of-fold calibration)
    abl: dict[str, dict] = {}
    for n, f in {**frames, **refs}.items():
        abl[f"reader: {n}" + (" (reference)" if n in refs else "")] = score(gold, reader_for_scoring(f, uids))
    public = {n: f for n, f in frames.items() if n != "claude"}
    abl["blend: public readers"] = score(gold, blend(public, uids, cfg)[0])
    abl["blend: public, no fills"] = score(gold, blend(public, uids, BlendConfig(weight_mode=a.weight_mode, fills={}))[0])
    abl["blend: public, silence=skip"] = score(gold, blend(public, uids, BlendConfig(weight_mode=a.weight_mode,
                                                                                     silence="skip"))[0])
    if "claude" in frames:
        prior_frame, _ = reader_frame(claude_long, gold, calibrate=False)
        abl["reader: claude (prior map)"] = score(gold, reader_for_scoring(prior_frame, uids))
        abl["blend: public + claude (prior map)"] = score(gold, blend({**public, "claude": prior_frame}, uids, cfg)[0])
    for n in frames:
        if len(frames) > 2:
            abl[f"blend without {n}"] = score(gold, blend({k: v for k, v in frames.items() if k != n}, uids, cfg)[0])
    abl["**final blend**"] = score(gold, p)

    per_finding = pd.DataFrame({"gold pos": gold.sum().astype(int)})
    for n, f in frames.items():
        per_finding[n] = per_finding_auc(gold.to_numpy(), reader_for_scoring(f, uids).reindex(gold.index).to_numpy())
    per_finding["blend"] = per_finding_auc(gold.to_numpy(), p.reindex(gold.index).to_numpy())
    per_finding.loc["MACRO"] = per_finding.mean()

    prev = pd.DataFrame({n: (f.reindex(nongold)[LABELS].fillna(0) > 0.5).mean() for n, f in frames.items()})
    prev["blend"] = (p.loc[nongold] > 0.5).mean()
    prev["gold 58"] = gold.mean()
    prev["blend mean p"] = p.loc[nongold].mean()

    stat = pd.DataFrame({"known": (status == "known").mean(), "fill": (status == "fill").mean(),
                         "silent": (status == "silent").mean(), "mean w": w.mean()})
    agree = spearman_matrix({**frames, "blend": p}, nongold)

    site = None
    sp = paths.audit_out / "study_train.parquet"
    if a.site_mode != "none" and sp.exists():
        site = pd.read_parquet(sp).set_index("StudyInstanceUID")["site_proxy"]
    folds = make_folds(train, p, seed=a.seed, site=site, site_mode=a.site_mode)
    fchk = check_folds(folds)
    fprev = pd.DataFrame({f: (p.loc[folds[folds.fold == f].StudyInstanceUID] > 0.5).mean()
                          for f in sorted(folds.fold.unique())})

    # ---- write the files
    lab = p.copy()
    for k in LABELS:
        lab[f"w_{k}"] = w[k]
    lab["is_gold"] = lab.index.isin(gold.index)
    lab.reset_index().to_parquet(out / f"labels_{a.version}.parquet", index=False)
    folds[["StudyInstanceUID", "fold"]].to_csv(out / f"folds_{a.version}.csv", index=False)
    folds.to_csv(out / f"folds_{a.version}_detail.csv", index=False)

    qa = {"version": a.version, "built": time.strftime("%Y-%m-%d %H:%M"), "readers": list(frames),
          "references": list(refs), "blend_config": cfg.__dict__, "calibrated": bool(cmaps) and not a.no_calibrate,
          "calibration_maps": cmaps, "ablations": abl, "leak_check": leaks, "folds": fchk,
          "site_mode": a.site_mode if site is not None else "none (no site proxy yet)", "cost": cost,
          "licenses": {k: v.get("license") for k, v in manifest.get("sources", {}).items()}}
    (out / f"qa_{a.version}.json").write_text(json.dumps(qa, indent=1, default=float), encoding="utf-8")
    write_disagreements(out / f"disagreements_{a.version}.md", train, gold, frames, p, claude_long)
    a.docs.write_text(render_md(qa, per_finding, prev, stat, agree, fprev, manifest), encoding="utf-8")
    print(f"wrote {out}/labels_{a.version}.parquet, folds_{a.version}.csv, qa, disagreements; {a.docs}")
    print(f"final blend gold macro AUC {_fmt(abl['**final blend**'])}; folds {fchk['sizes']} gold {fchk['gold_per_fold']}")


def write_disagreements(path: Path, train: pd.DataFrame, gold: pd.DataFrame, frames: dict, p: pd.DataFrame,
                        claude_long: pd.DataFrame | None, n: int = 40) -> None:
    """(a) gold cells the blend gets most wrong; (b) non-gold cells where readers disagree most. With quotes."""
    rep = train.set_index("StudyInstanceUID").Report
    q = {}
    if claude_long is not None:
        q = {(r.StudyInstanceUID, r.label): f"{r.state}/{r.severity}{' deg' if r.degenerative else ''}: {r.quote}"
             for r in claude_long.itertuples()}
    L = ["# Label disagreements (MIRA-derived: contains report text; never commit)", ""]
    rows = []
    for u in gold.index:
        for k in LABELS:
            rows.append((abs(p.loc[u, k] - gold.loc[u, k]), u, k))
    rows.sort(reverse=True)
    L += ["## A. Gold cells the blend gets most wrong", ""]
    for d, u, k in rows[:n]:
        vals = ", ".join(f"{nm}={frames[nm].reindex([u])[k].iloc[0]:.2f}" for nm in frames)
        L += [f"### {k} — gold {int(gold.loc[u, k])}, blend {p.loc[u, k]:.2f} — …{u[-10:]}",
              f"- readers: {vals}", f"- claude: {q.get((u, k), 'n/a')}",
              f"- report: {rep[u][:1200]}", ""]
    names = list(frames)
    R = np.stack([frames[nm].reindex(p.index)[LABELS].to_numpy(float) for nm in names])
    spread = np.nanmax(R, axis=0) - np.nanmin(R, axis=0)
    spread[np.isfinite(R).sum(axis=0) < 2] = -1
    flat = np.argsort(-np.nan_to_num(spread, nan=-1).ravel())[:n]
    L += ["## B. Non-gold cells where the readers disagree most", ""]
    for i in flat:
        r, c = divmod(int(i), len(LABELS))
        u, k = p.index[r], LABELS[c]
        vals = ", ".join(f"{nm}={R[j, r, c]:.2f}" for j, nm in enumerate(names))
        L += [f"### {k} — blend {p.loc[u, k]:.2f} — …{u[-10:]}", f"- readers: {vals}",
              f"- claude: {q.get((u, k), 'n/a')}", f"- report: {rep[u][:1200]}", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def render_md(qa, per_finding, prev, stat, agree, fprev, manifest) -> str:
    L = [f"# Labels {qa['version']}: QA report", "",
         f"_Generated by `python -m knee.labels.build_labels` on {qa['built']}. Numbers only; no report text "
         f"(the disagreement list with quotes is `<cache_root>/labels/disagreements_{qa['version']}.md`, gitignored)._",
         "", "## 1. Sources", "",
         "| reader | role | license | note |", "|---|---|---|---|"]
    for s in PS.SOURCES:
        lic = ", ".join(manifest.get("sources", {}).get(s.name, {}).get("license") or ["?"])
        role = "reader (blended)" if s.name in qa["readers"] else s.role
        L.append(f"| {s.name} | {role} | {lic} | {s.note} |")
    if "claude" in qa["readers"]:
        L.append(f"| claude | reader (blended) | ours | Claude API reader, prompt {__import__('knee.labels.prompt', fromlist=['x']).PROMPT_VERSION}, "
                 f"{'gold-calibrated (5-fold OOF)' if qa['calibrated'] else 'prior map'} |")
    L += ["", "Excluded without download: yunusgmsoy v5 and barun2104 (gold labels pasted into the 58 rows).",
          "Leak check (≥10 findings with AUC ≥ 0.999, or ≥95% of gold rows reproduced): "
          + ", ".join(f"{k} {'LEAK' if v['leaked'] else 'ok'}" for k, v in qa["leak_check"].items()) + ".", "",
          "## 2. Gold-58 macro AUC (bootstrap 95% CI, 1,000 resamples)", "",
          "Single readers score silent cells as 0.5. Nothing here is tuned on gold except our reader's calibration, "
          "which is 5-fold out-of-fold. Differences under ~0.02 are not measurable on 58 studies.", "",
          "| variant | gold macro AUC [95% CI] |", "|---|---|"]
    L += [f"| {k} | {_fmt(v)} |" for k, v in qa["ablations"].items()]
    L += ["", "## 3. Per-finding gold AUC", "", df_to_markdown(per_finding.reset_index(names="finding"), ".3f"),
          "", "## 4. Prevalence on the 4,349 non-gold studies (share with p > 0.5; silent = negative)", "",
          df_to_markdown(prev.reset_index(names="finding"), ".3f"),
          "", "Gold prevalence is the annotators' sampling (every gold study has ≥1 positive), not disease prevalence.",
          "", "## 5. Blend cell status and mean weight", "",
          df_to_markdown(stat.reset_index(names="finding"), ".3f"),
          "", "known = at least one reader spoke; fill = co-occurrence fill (Synovitis ← Effusion, PF OA ← Medial OA); "
          "silent = nobody spoke → p = silent_p at low weight.", "",
          "## 6. Reader agreement (mean Spearman over findings, non-gold, cells where both spoke)", "",
          df_to_markdown(agree.reset_index(names="reader"), ".3f"), "",
          "## 7. Folds", "",
          f"- Sizes {qa['folds']['sizes']}; gold per fold {qa['folds']['gold_per_fold']}; "
          f"{qa['folds']['n_groups']} groups, {qa['folds']['multi_study_groups']} with >1 study (shared report text).",
          f"- Site handling: {qa['site_mode']}.", "",
          "Blended prevalence (p > 0.5) per fold:", "",
          df_to_markdown(fprev.reset_index(names="finding"), ".3f"), ""]
    if qa["calibrated"]:
        L += ["## 8. Our reader's calibration maps (fitted on all 58; gold rows used 5-fold out-of-fold values)", "",
              "| finding | " + " | ".join(qa["calibration_maps"][LABELS[0]].keys()) + " |",
              "|---|" + "---|" * len(qa["calibration_maps"][LABELS[0]])]
        for k, mp in qa["calibration_maps"].items():
            L.append(f"| {k} | " + " | ".join(f"{v:.2f}" for v in mp.values()) + " |")
        L.append("")
    L += ["## 9. API cost", "", f"```\n{json.dumps(qa['cost'], indent=1) if qa['cost'] else 'no Claude run yet'}\n```", ""]
    return "\n".join(L)


if __name__ == "__main__":
    main()
