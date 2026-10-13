"""S04 label pipeline tests (CPU, synthetic data, no network, no API key): prompt/schema, reader bookkeeping and cost,
public-set parsing + leak check, blend semantics, out-of-fold calibration, folds, and an end-to-end build."""
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pytest

from knee.labels import prompt as P
from knee.labels import public_sets as PS
from knee.labels import reader as RD
from knee.labels.blend import BlendConfig, blend
from knee.labels.calibrate import fit_map, reader_frame
from knee.labels.folds import check_folds, make_folds
from knee.paths import LABELS


def _response(states: dict | None = None) -> dict:
    f = {}
    for k in P.KEYS.values():
        st, sev, deg = (states or {}).get(k, ("not_mentioned", 0, False))
        f[k] = {"state": st, "severity": sev, "degenerative": deg, "postoperative": False,
                "quote": "" if st == "not_mentioned" else "q"}
    return {"language": "English", "knee_side": "left", "findings": f}


# ------------------------------------------------------------------------------------------ prompt
def test_schema_is_strict_everywhere():
    def walk(s):
        if s.get("type") == "object":
            assert s["additionalProperties"] is False
            assert set(s["required"]) == set(s["properties"])
            for v in s["properties"].values():
                walk(v)
    walk(P.output_schema())
    assert list(P.output_schema()["properties"]["findings"]["properties"]) == list(P.KEYS.values())


def test_parse_output_and_bins():
    d = P.parse_output(json.dumps(_response({"effusion": ("present", 1, False)})))
    assert d["findings"]["effusion"]["severity"] == 1
    with pytest.raises(ValueError):
        bad = _response()
        del bad["findings"]["acl"]
        P.parse_output(json.dumps(bad))
    assert P.to_bin("not_mentioned", 0, False) is None
    assert P.to_bin("absent", 0, True) == "absent_degenerative"
    assert P.to_bin("present", 0, False) == "present_1"
    assert P.to_bin("present", 3, False) == "present_3"
    assert np.isnan(P.prior_prob("ACL", None))
    assert P.prior_prob("Effusion", "present_1") < P.prior_prob("Fracture", "present_1")
    # the prior is monotone in the bin order for every finding
    for k in LABELS:
        v = [P.prior_prob(k, b) for b in P.BINS]
        assert v == sorted(v), k


# ------------------------------------------------------------------------------------------ reader bookkeeping
def test_custom_id_and_request_params():
    cid = RD.custom_id("1.2.826.0.1.3680043.8.498.100")
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", cid) and cid == RD.custom_id("1.2.826.0.1.3680043.8.498.100")
    rp = RD.request_params("Knee MRI. ACL intact.", "claude-opus-5", "low", "adaptive", 4000, "1h")
    assert rp["system"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}
    assert rp["output_config"]["format"]["type"] == "json_schema" and rp["output_config"]["effort"] == "low"
    assert rp["thinking"] == {"type": "adaptive"}
    assert "<report>" in rp["messages"][0]["content"]


def test_cost_arithmetic():
    rec = {"model": "claude-opus-5", "mode": "batch",
           "usage": {"input_tokens": 1_000_000, "output_tokens": 1_000_000,
                     "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1_000_000}}
    # (5 + 25 + 0.1·5) × 0.5 batch discount
    assert RD.cost_of(rec) == pytest.approx((5 + 25 + 0.5) * 0.5)
    assert RD.cost_of({**rec, "mode": "sync"}) == pytest.approx(30.5)
    assert RD.cost_of({**rec, "model": "unknown"}) == 0.0


def test_run_done_and_parse(tmp_path):
    run = RD.Run(tmp_path / "r")
    ok = RD._record("u1", "claude-opus-5", "batch")
    ok.update(text=json.dumps(_response({"acl": ("present", 3, False)})), usage={"input_tokens": 10, "output_tokens": 5,
              "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0})
    trunc = {**RD._record("u2", "claude-opus-5", "batch", error="max_tokens"), "text": '{"language": "En'}
    old = {**ok, "uid": "u3", "prompt_version": "v0"}
    for r in (ok, trunc, old):
        run.save_raw(RD.custom_id(r["uid"]), r)
    assert run.done_uids() == {"u1"}  # truncated JSON and an old prompt version are redone
    long = RD.parse_run(run)
    assert len(long) == 12 and set(long.label) == set(LABELS)
    assert long.set_index("label").loc["ACL", "bin"] == "present_3"
    assert pd.isna(long.set_index("label").loc["MCL", "bin"])
    s = RD.cost_summary(run, n_total=100)
    assert s["done_studies"] == 1 and s["errors"] == 1


def test_pilot_selection_is_gold_plus_random_and_deterministic():
    rng = np.random.default_rng(0)
    langs = ["The knee has no effusion and the ligaments are intact.",
             "La rodilla sin derrame y los ligamentos de la articulacion.",
             "Diz eklem ile sıvı yok ve bir değişiklik izlendi olarak."]
    df = pd.DataFrame({"StudyInstanceUID": [f"u{i}" for i in range(300)],
                       "Report": [langs[i % 3] + f" {i}" for i in range(300)],
                       "is_gold": rng.random(300) < 0.1})
    a, b = RD.pilot_uids(df, 42, 0), RD.pilot_uids(df, 42, 0)
    assert a == b and len(set(a)) == len(a) == df.is_gold.sum() + 42
    assert set(df[df.is_gold].StudyInstanceUID) <= set(a)


# ------------------------------------------------------------------------------------------ public sets
def _std(values: dict, uids) -> pd.DataFrame:
    return pd.DataFrame({k: values.get(k, np.nan) for k in LABELS}, index=pd.Index(uids, name="StudyInstanceUID"))


def test_public_parsers_turn_silence_into_nan():
    uids = ["a", "b"]
    pk = pd.DataFrame({"StudyInstanceUID": uids, **{k: [0.08, 0.28] for k in LABELS},
                       **{f"{k}__verdict": ["NO", "UNK"] for k in LABELS}})
    out = PS._pilkwang(pk)
    assert out.loc["a", "ACL"] == 0.08 and np.isnan(out.loc["b", "ACL"])
    st = pd.DataFrame({"StudyInstanceUID": uids, **{k: [0.5, 0.9] for k in LABELS}})
    assert np.isnan(PS._steven(st).loc["a", "MCL"]) and PS._steven(st).loc["b", "MCL"] == 0.9
    lm = pd.DataFrame({"StudyInstanceUID": uids, **{f"{k}__confidence": [0.1, 0.9] for k in LABELS},
                       **{f"{k}__mentioned": [0, 1] for k in LABELS}})
    assert np.isnan(PS._laymond(lm).loc["a", "PF OA"]) and PS._laymond(lm).loc["b", "PF OA"] == 0.9


def test_leak_check_flags_pasted_gold():
    rng = np.random.default_rng(1)
    uids = [f"g{i}" for i in range(58)]
    gold = pd.DataFrame(rng.integers(0, 2, (58, 12)).astype(float), index=uids, columns=LABELS)
    assert PS.leak_check(gold.copy(), gold)["leaked"]
    noisy = (gold * 0.6 + rng.random((58, 12)) * 0.4)
    noisy.iloc[:20] = rng.random((20, 12))
    assert not PS.leak_check(noisy, gold)["leaked"]


# ------------------------------------------------------------------------------------------ blend
def test_blend_silence_semantics():
    uids = ["s1", "s2", "s3"]
    a = _std({"ACL": [0.05, np.nan, np.nan], "Effusion": [0.9, 0.9, np.nan]}, uids)
    b = _std({"ACL": [np.nan, np.nan, np.nan], "Effusion": [0.9, 0.9, np.nan]}, uids)
    lix = _std({k: [0.0, 0.0, 0.0] for k in LABELS}, uids)  # never silent: 0 = silence in disguise
    lix.loc["s3", "Fracture"] = 1.0
    p, w, st = blend({"pilkwang": a, "steven_v1": b, "lixin": lix}, uids, BlendConfig())
    # s1 ACL: a says 0.05, b silent (votes 0.5), lixin 0 → known, mean = 0.183
    assert st.loc["s1", "ACL"] == "known" and p.loc["s1", "ACL"] == pytest.approx((0.05 + 0.5 + 0.0) / 3)
    # s2 ACL: only lixin's 0 → silent, not a confident negative
    assert st.loc["s2", "ACL"] == "silent" and p.loc["s2", "ACL"] == 0.5 and w.loc["s2", "ACL"] == 0.10
    # s3 Fracture: lixin asserts a positive → known
    assert st.loc["s3", "Fracture"] == "known" and p.loc["s3", "Fracture"] > 0.5
    # Synovitis silent everywhere → filled from Effusion (0.22 + 0.41·p_eff)
    assert st.loc["s1", "Synovitis"] == "fill" and p.loc["s1", "Synovitis"] > p.loc["s3", "Synovitis"]
    p2, _, _ = blend({"pilkwang": a, "steven_v1": b, "lixin": lix}, uids, BlendConfig(silence="skip"))
    assert p2.loc["s1", "ACL"] == pytest.approx((0.05 + 0.0) / 2)
    assert ((w >= 0) & (w <= 1)).all().all() and ((p >= 0) & (p <= 1)).all().all()
    # v1 (E016): silent cells become a soft negative; known and filled cells are unchanged
    p3, w3, _ = blend({"pilkwang": a, "steven_v1": b, "lixin": lix}, uids, BlendConfig(silent_cell_p=0.15))
    assert p3.loc["s2", "ACL"] == 0.15 and w3.loc["s2", "ACL"] == 0.50
    assert p3.loc["s1", "ACL"] == p.loc["s1", "ACL"] and p3.loc["s1", "Synovitis"] == p.loc["s1", "Synovitis"]


def test_blend_weight_rewards_agreement():
    uids = ["x", "y"]
    a = _std({"ACL": [0.95, 0.95]}, uids)
    b = _std({"ACL": [0.95, 0.05]}, uids)
    _, w, _ = blend({"pilkwang": a, "steven_v1": b}, uids)
    assert w.loc["x", "ACL"] > w.loc["y", "ACL"]


# ------------------------------------------------------------------------------------------ calibration
def test_fit_map_shrinks_and_is_monotone():
    empty = fit_map(pd.Series([], dtype=object), pd.Series([], dtype=float), "ACL")
    assert empty == pytest.approx({b: P.prior_prob("ACL", b) for b in P.BINS})
    bins = pd.Series(["present_1"] * 20 + ["uncertain"] * 20)
    y = pd.Series([0.0] * 20 + [1.0] * 20)  # data says mild < uncertain: isotonic must not reorder
    mp = fit_map(bins, y, "Effusion")
    v = [mp[b] for b in P.BINS]
    assert v == sorted(v)
    assert mp["present_1"] == pytest.approx(mp["uncertain"])  # pooled


def test_calibration_is_out_of_fold_for_gold():
    uids = [f"g{i}" for i in range(6)]
    long = pd.DataFrame([{"StudyInstanceUID": u, "label": k,
                          "bin": ("present_3" if (u == "g0" and k == "ACL") else "absent")} for u in uids for k in LABELS])
    gold = pd.DataFrame(0.0, index=uids, columns=LABELS)
    gold.loc["g0", "ACL"] = 1.0
    frame, maps = reader_frame(long, gold, calibrate=True, m=5.0, oof="loo")
    # all-gold map sees g0's positive in present_3; g0's own LOO value can't (that bin is empty without it → prior)
    assert maps["ACL"]["present_3"] > P.prior_prob("ACL", "present_3")
    assert frame.loc["g0", "ACL"] == pytest.approx(P.prior_prob("ACL", "present_3"))
    prior, _ = reader_frame(long, gold, calibrate=False)
    assert prior.loc["g1", "ACL"] == P.prior_prob("ACL", "absent")
    kf, _ = reader_frame(long, gold, calibrate=True, m=5.0, oof=3)  # K-fold: g0 is never in its own fitting fold
    assert kf.loc["g0", "ACL"] == pytest.approx(P.prior_prob("ACL", "present_3"))
    from knee.labels.calibrate import gold_folds
    folds = gold_folds(uids, 3, seed=1)
    assert sorted(u for f in folds for u in f) == sorted(uids) and len(folds) == 3


# ------------------------------------------------------------------------------------------ folds
def _train(n=400, n_gold=40, seed=0):
    rng = np.random.default_rng(seed)
    reports = [f"report number {i}" for i in range(n)]
    for i in range(0, 60, 3):  # duplicate report groups (case/whitespace variants must still group)
        reports[i + 1] = reports[i].upper() + "  "
        reports[i + 2] = reports[i]
    df = pd.DataFrame({"StudyInstanceUID": [f"u{i}" for i in range(n)], "Report": reports})
    for k in LABELS:
        df[k] = np.nan
    gi = rng.choice(np.arange(60, n), n_gold, replace=False)
    df.loc[gi, LABELS] = rng.integers(0, 2, (n_gold, 12)).astype(float)
    p = pd.DataFrame(rng.random((n, 12)), index=df.StudyInstanceUID, columns=LABELS)
    return df, p


def test_folds_group_duplicates_and_spread_gold():
    df, p = _train()
    f = make_folds(df, p, n_folds=5, seed=0)
    chk = check_folds(f)
    assert sum(chk["sizes"]) == len(df) and max(chk["sizes"]) - min(chk["sizes"]) <= 10
    assert min(chk["gold_per_fold"]) >= 7 and max(chk["gold_per_fold"]) <= 9
    ff = f.set_index("StudyInstanceUID").fold
    for i in range(0, 60, 3):
        assert ff[f"u{i}"] == ff[f"u{i + 1}"] == ff[f"u{i + 2}"]
    assert f.equals(make_folds(df, p, n_folds=5, seed=0))


def test_folds_site_group_mode_holds_out_whole_sites():
    df, p = _train()
    site = pd.Series([f"site{(i // 3) % 10}" for i in range(len(df))], index=df.StudyInstanceUID)  # dup triples share a site
    f = make_folds(df, p, n_folds=5, seed=0, site=site, site_mode="group")
    per_site = f.assign(site=site.to_numpy()).groupby("site").fold.nunique()
    assert (per_site == 1).all()


# ------------------------------------------------------------------------------------------ end to end
def test_build_labels_end_to_end(tmp_path, monkeypatch):
    df, _ = _train(n=300, n_gold=30, seed=3)
    data, cache = tmp_path / "data", tmp_path / "cache"
    data.mkdir()
    df.to_csv(data / "train.csv", index=False)
    rng = np.random.default_rng(3)
    uids = df.StudyInstanceUID
    truth = rng.random((len(df), 12))
    root = PS.public_dir(cache)
    for s in PS.SOURCES:
        d = root / s.slug.replace("/", "_")
        d.mkdir(parents=True, exist_ok=True)
        noisy = np.clip(truth + rng.normal(0, 0.2, truth.shape), 0, 1).round(2)
        if s.name == "pilkwang":
            out = pd.DataFrame({"StudyInstanceUID": uids})
            for j, k in enumerate(LABELS):
                out[k] = noisy[:, j]
                out[f"{k}__verdict"] = np.where(rng.random(len(df)) < 0.2, "UNK", "YES")
        elif s.name == "laymond":
            out = pd.DataFrame({"StudyInstanceUID": uids})
            for j, k in enumerate(LABELS):
                out[f"{k}__confidence"] = noisy[:, j]
                out[f"{k}__mentioned"] = (rng.random(len(df)) > 0.2).astype(int)
        else:
            out = pd.DataFrame(noisy, columns=LABELS)
            out.insert(0, "StudyInstanceUID", uids)
        if not (d / s.file).exists():
            out.to_csv(d / s.file, index=False)
    for k, v in {"KNEE_DATA_ROOT": data, "KNEE_CACHE_ROOT": cache, "KNEE_AUDIT_OUT": tmp_path / "audit"}.items():
        monkeypatch.setenv(k, str(v))
    import knee.labels.build_labels as BL
    monkeypatch.setattr(BL, "N_BOOT", 20)
    docs = tmp_path / "labels.md"
    monkeypatch.setattr("sys.argv", ["build_labels", "--version", "t", "--no-claude", "--docs", str(docs)])
    BL.main()
    lab = pd.read_parquet(cache / "labels" / "labels_t.parquet")
    assert list(lab.columns[:13]) == ["StudyInstanceUID"] + LABELS
    assert {f"w_{k}" for k in LABELS} <= set(lab.columns) and lab.is_gold.sum() == 30
    assert lab[LABELS].notna().all().all() and len(lab) == len(df)
    folds = pd.read_csv(cache / "labels" / "folds_t.csv")
    assert list(folds.columns) == ["StudyInstanceUID", "fold"] and set(folds.fold) == set(range(5))
    md = docs.read_text(encoding="utf-8")
    assert "final blend" in md and "report number" not in md  # no report text in the committed doc
    assert "report number" in (cache / "labels" / "disagreements_t.md").read_text(encoding="utf-8")


def test_submit_collect_with_fake_client(tmp_path, monkeypatch):
    """The batch path end to end with a fake SDK client: pilot submit → collect → parse, then resubmission skips
    finished studies, and the full run refuses to go over budget."""
    import types
    anthropic = pytest.importorskip("anthropic")
    df, _ = _train(n=120, n_gold=10, seed=5)
    reports = df.assign(is_gold=df[LABELS].notna().all(axis=1))[["StudyInstanceUID", "Report", "is_gold"]]
    store: dict = {}

    class Batches:
        def create(self, requests):
            bid = f"b{len(store)}"
            store[bid] = [r["custom_id"] for r in requests]
            assert all(r["params"]["model"] == "claude-opus-5" for r in requests)
            return types.SimpleNamespace(id=bid)

        def retrieve(self, bid):
            return types.SimpleNamespace(processing_status="ended", request_counts={"succeeded": len(store[bid])})

        def results(self, bid):
            for cid in store[bid]:
                msg = types.SimpleNamespace(
                    stop_reason="end_turn", model="claude-opus-5",
                    content=[types.SimpleNamespace(type="text", text=json.dumps(_response({"acl": ("absent", 0, False)})))],
                    usage=types.SimpleNamespace(input_tokens=600, output_tokens=700, cache_creation_input_tokens=0,
                                                cache_read_input_tokens=2000))
                yield types.SimpleNamespace(custom_id=cid, result=types.SimpleNamespace(type="succeeded", message=msg))

    fake = types.SimpleNamespace(messages=types.SimpleNamespace(batches=Batches()))
    monkeypatch.setattr(anthropic, "Anthropic", lambda: fake)
    args = types.SimpleNamespace(split="pilot", n_random=20, seed=0, limit=0, budget=50.0, chunk=5000,
                                 model="claude-opus-5", effort="low", thinking="adaptive", max_tokens=4000)
    run = RD.Run(tmp_path / "run")
    RD.cmd_submit(args, run, reports)
    assert len(run.batches()) == 1 and len(run.batches()[0]["uids"]) == 10 + 20
    RD.cmd_submit(args, run, reports)  # pending → nothing new
    assert len(run.batches()) == 1
    RD.cmd_collect(args, run, reports)
    assert len(run.done_uids()) == 30 and run.batches()[0]["collected"]
    per = RD.cost_summary(run, len(reports))["usd_per_report_batch"]
    assert per == pytest.approx((600 * 5 + 2000 * 0.5 + 700 * 25) / 1e6 * 0.5)
    RD.cmd_submit(types.SimpleNamespace(**{**vars(args), "split": "full"}), run, reports)
    assert len(run.batches()[-1]["uids"]) == 120 - 30  # only the unfinished studies
    with pytest.raises(SystemExit):
        run2 = RD.Run(tmp_path / "run2")
        for f in run.raw.glob("r*.json"):
            run2.save_raw(f.stem, json.loads(f.read_text(encoding="utf-8")))
        RD.cmd_submit(types.SimpleNamespace(**{**vars(args), "split": "full", "budget": 0.001}), run2, reports)
    assert len(RD.parse_run(run)) == 30 * 12


def test_agent_reader_export_ingest_roundtrip(tmp_path):
    from knee.labels import agent_reader as AR
    rng = np.random.default_rng(0)
    n = 30
    df = pd.DataFrame({"StudyInstanceUID": [f"1.2.{i}" for i in range(n)],
                       "Report": [f"report {i} ACL intact" for i in range(n)]})
    for k in LABELS:
        df[k] = np.where(np.arange(n) < 5, rng.integers(0, 2, n), np.nan)
    csv = tmp_path / "train.csv"
    df.to_csv(csv, index=False)
    reports = RD.load_reports(csv)
    run = RD.Run(tmp_path / "claude" / "agent_t")

    class A:
        split, n_random, seed, shard_size, model = "full", 0, 0, 12, "claude-opus-5-5"
    AR.cmd_export(A, run, reports)
    shards = sorted((run.root / "shards").glob("s*.jsonl"))
    assert [s.stem for s in shards] == ["s0000", "s0001", "s0002"]
    ids = [json.loads(x)["id"] for s in shards for x in s.read_text(encoding="utf-8").splitlines()]
    assert sorted(ids) == sorted(RD.custom_id(u) for u in df.StudyInstanceUID)  # every study once
    assert ids[:5] != [RD.custom_id(u) for u in df.StudyInstanceUID[:5]]  # shuffled: gold not first
    assert "uid" not in shards[0].read_text(encoding="utf-8") and "1.2." not in shards[0].read_text(encoding="utf-8")  # blinded

    first = [json.loads(x)["id"] for x in shards[0].read_text(encoding="utf-8").splitlines()]
    lines = [json.dumps({"id": c, "language": "English", "knee_side": "left",
                         "f": {"acl": ["absent", 0, False, False, "ACL intact"]}}) for c in first[:-2]]
    lines.append(json.dumps({"id": first[-2], "f": {"acl": ["torn", 3, False, False, ""]}}))  # bad state
    (run.root / "out" / "s0000.jsonl").write_text("\n".join(lines) + "\nnot json\n")
    AR.cmd_ingest(A, run, reports)
    log = json.loads((run.root / "ingest_log.json").read_text(encoding="utf-8"))["s0000"]
    assert log["accepted"] == 10 and len(log["rejected"]) == 2 and len(log["missing"]) == 2
    assert len(run.done_uids()) == 10
    long = RD.parse_run(run)
    acl = long[long.label == "ACL"]
    assert (acl.bin == "absent").all() and long[long.label == "MCL"].bin.isna().all()  # omitted = not_mentioned
    AR.cmd_export(A, run, reports)  # s0001/s0002 still open, s0000's 2 failures re-exported
    new = sorted((run.root / "shards").glob("s*.jsonl"))[-1]
    assert new.stem == "s0003" and len(new.read_text(encoding="utf-8").splitlines()) == 2
    assert "AGENT_INSTRUCTIONS.md" in AR.dispatch_prompt(run.root, ["s0001"])


def test_agent_reader_scan_split_excludes_gold_and_pilot():
    from knee.labels import agent_reader as AR
    rng = np.random.default_rng(1)
    n = 300
    df = pd.DataFrame({"StudyInstanceUID": [f"1.3.{i}" for i in range(n)],
                       "Report": [f"report {i}: the ACL is intact" for i in range(n)]})
    for k in LABELS:
        df[k] = np.where(np.arange(n) < 20, rng.integers(0, 2, n), np.nan)
    df["is_gold"] = np.arange(n) < 20
    scan = AR.scan_uids(df, n=50)
    assert len(scan) == len(set(scan)) == 50
    assert not set(scan) & set(RD.pilot_uids(df, 42, 0))  # pilot and gold excluded
    assert AR.scan_uids(df, n=50) == scan  # seeded
    assert "up to 10 short notes" in AR.dispatch_prompt(Path("/r"), ["s0000"], scan=True)
    assert "at most 3 short notes" in AR.dispatch_prompt(Path("/r"), ["s0000"])
