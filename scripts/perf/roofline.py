import json, glob, collections
import os
S = os.environ.get("PERF_OUT", ".")
# peak half-precision tensor FLOP/s (fp32 accumulate), DRAM bytes/s. Achieved fractions are assumptions.
HW = {"T4": (65e12, 320e9), "5070": (62e12, 672e9)}
EFF_C, EFF_B = 0.5, 0.75       # achieved fraction of peak for GEMMs / for bandwidth-bound kernels
LAUNCH = 10e-6                 # eager PyTorch CPU cost per aten op (dispatch + launch), UNVERIFIED on Windows
res = []
for f in sorted(glob.glob(f"{S}/ops_*.json")):
    res += json.load(open(f, encoding="utf-8"))
for r in res:
    print(f"\n=== {r['tag']}")
    ops = r["ops"]
    n = sum(o[2] for o in ops)
    F = sum(o[3] for o in ops); B = sum(o[4] for o in ops)
    by = collections.defaultdict(lambda: [0, 0, 0])
    for c, name, k, fl, by_ in ops:
        by[c][0] += k; by[c][1] += fl; by[c][2] += by_
    print(f"aten ops (≈kernel launches) {n}, FLOPs {F/1e12:.2f} T, bytes {B/1e9:.1f} GB, intensity {F/B:.0f} FLOP/B")
    for hw, (pk, bw) in HW.items():
        t = collections.Counter()
        for c, name, k, fl, by_ in ops:
            tc, tb = fl / (pk * EFF_C), by_ / (bw * EFF_B)
            t["compute-bound" if tc >= tb else ("cast" if c == "cast" else "bandwidth-bound")] += max(tc, tb)
        gpu = sum(t.values()); cpu = n * LAUNCH
        print(f"  {hw}: GPU {gpu*1e3:7.1f} ms  = " + ", ".join(f"{k} {v/gpu:4.0%}" for k, v in t.most_common())
              + f" | CPU dispatch {cpu*1e3:.1f} ms ({'OVERHEAD-BOUND' if cpu > gpu else 'GPU-bound'})")
    top = sorted(ops, key=lambda o: -max(o[3] / (65e12 * EFF_C), o[4] / (320e9 * EFF_B)))[:12]
    print("  top ops on T4 (name, cat, calls, GFLOP, GB, est ms):")
    for c, name, k, fl, by_ in top:
        print(f"    {name:42s} {c:5s} {k:5d} {fl/1e9:9.1f} {by_/1e9:7.2f} {max(fl/(65e12*EFF_C), by_/(320e9*EFF_B))*1e3:7.1f}")
