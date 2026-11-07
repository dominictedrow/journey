import time
import numpy as np


def time_fn(fn, n_runs=100, warmup=10):
    times = []
    for i in range(n_runs+warmup):
        if i < warmup:
            fn()
            continue
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append(end-start)
    return np.array(times)

def summarize(times):
    times = times *1000
    p50,p99,p95 = np.percentile(times, [50,99,95])
    out = {'p50': p50, 'p95': p95, 'p99': p99, 'mean': np.mean(times), 'max': np.max(times)}
    return out

def benchmark_batches(predict_fn, input_dim, batch_sizes=(1, 8, 64), n_runs=100, warmup=10, seed=0):
    batches = {}
    rng = np.random.default_rng(seed)
    for i in batch_sizes:
        x = rng.random((i,input_dim))
        times = time_fn(lambda: predict_fn(x), n_runs, warmup)
        summary = summarize(times)
        summary.setdefault('throughput', i/(summary['p50']/1000))
        batches.setdefault(i,summary)

    return batches

def compare_fns(fns, n_runs=20, warmup=2):
    final = {}
    for i in fns:
        times = time_fn(fns[i], n_runs, warmup)
        summary = summarize(times)     
        final.setdefault(i,summary)
    fast = min(s['p50'] for s in final.values())
    for i in final.values():
        i.setdefault('slowdown',(i['p50']/fast))
    return final




