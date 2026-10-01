"""Stage the Kaggle "bundle" dataset folder that notebooks/infer.py reads (S06 steps 1–3). Never uploads anything.

    python scripts/build_kaggle_bundle.py --out <dir> --ckpt runs/trackA_f0/model_avg.pt [--ckpt ... --weight 1 ...]
        --cache-manifest <cache_dir>/cache_manifest.json [--labels <cache_root>/labels/labels_v1.parquet]
        [--wheels <dir of .whl>] [--how mean] [--kaggle-user <name>]

Result:
    <out>/knee_src/knee/...   copy of src/knee (no __pycache__, no ._* AppleDouble files)
    <out>/weights/arm<i>.pt   the checkpoints
    <out>/wheels/*.whl        optional; build them for the Kaggle image's Python (3.12.13, verified by knee-probe 2026-09-29), e.g. in an
                              internet-on Kaggle notebook: pip download pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg
                              python-gdcm --dest wheels
    <out>/arms.json           arms + blend + fallback prior (mean soft label of the report labels, else 0.5)
    <out>/dataset-metadata.json  for `kaggle datasets create -p <out> --dir-mode zip` — **run only with Dom's OK**

Keep <out> off the FAT32 project drive when checkpoints are large (4 GB per-file limit).
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def preproc_from_manifest(man: dict) -> dict:
    """Inference PreprocParams from a cache manifest. A derived cache (knee.data.derive_cache) stores the source
    cache's params with out_px replaced; its windows are source-resolution renders resized with INTER_AREA. Inference
    must render at the source out_px and let the arm's data.out_px do the same resize at load, so restore it here
    (rendering straight at the derived out_px is a different resampling path; review 2026-09-30)."""
    pp = dict(man["params"])
    src = man.get("derived_from")
    if src:
        pp["out_px"] = int(src["src_out_px"])
    return pp


def build(out: Path, ckpts: list[Path], weights: list[float], how: str, labels: Path | None, wheels: Path | None,
          kaggle_user: str, slug: str, preproc: dict) -> dict:
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from knee.paths import LABELS

    out.mkdir(parents=True, exist_ok=True)
    src = out / "knee_src" / "knee"
    if src.exists():
        shutil.rmtree(src)
    shutil.copytree(REPO / "src" / "knee", src, ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "._*"))
    (out / "weights").mkdir(exist_ok=True)
    arms = []
    for i, (c, w) in enumerate(zip(ckpts, weights)):
        dst = out / "weights" / f"arm{i}.pt"
        shutil.copy2(c, dst)
        arms.append({"ckpt": f"weights/arm{i}.pt", "weight": w, "source": str(c)})
    prior = [0.5] * len(LABELS)
    if labels is not None:
        import pandas as pd
        prior = pd.read_parquet(labels)[LABELS].mean().round(4).tolist()
    if wheels is not None:
        (out / "wheels").mkdir(exist_ok=True)
        for whl in Path(wheels).glob("*.whl"):
            if whl.name.startswith("numpy-"):  # pip download adds it as a dependency; the image's numpy must stay
                continue
            shutil.copy2(whl, out / "wheels" / whl.name)
    spec = {"arms": arms, "how": how, "prior": prior, "preproc": preproc}
    (out / "arms.json").write_text(json.dumps(spec, indent=1))
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": slug, "id": f"{kaggle_user}/{slug}", "licenses": [{"name": "CC0-1.0"}]}, indent=1))
    return spec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ckpt", type=Path, action="append", required=True)
    ap.add_argument("--weight", type=float, action="append")
    ap.add_argument("--how", default="mean", choices=["mean", "rank"])
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--wheels", type=Path)
    ap.add_argument("--cache-manifest", type=Path,
                    help="the training cache's cache_manifest.json: its params become the inference PreprocParams "
                         "(train/infer parity). Without it the code defaults are used, with a warning.")
    ap.add_argument("--kaggle-user", default="dominictedrow")
    ap.add_argument("--slug", default="knee-bundle")
    a = ap.parse_args()
    weights = a.weight or [1.0] * len(a.ckpt)
    if len(weights) != len(a.ckpt):
        raise SystemExit("--weight must be given once per --ckpt (or not at all)")
    if a.cache_manifest:
        preproc = preproc_from_manifest(json.loads(a.cache_manifest.read_text()))
    else:
        preproc = {}
        print("WARNING: no --cache-manifest: inference uses the PreprocParams defaults; make sure the cache did too")
    spec = build(a.out, a.ckpt, weights, a.how, a.labels, a.wheels, a.kaggle_user, a.slug, preproc)
    print(json.dumps(spec, indent=1))
    print(f"\nstaged {a.out}. Upload (only with Dom's OK): kaggle datasets create -p {a.out} --dir-mode zip")


if __name__ == "__main__":
    main()
