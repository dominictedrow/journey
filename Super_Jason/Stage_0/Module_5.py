import time
import numpy as np


def time_fn(fn, n_runs=100, warmup=10):
    times = []
    for i in range(n_runs+warmup):
        if i < 10:
            continue
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append(end-start)
    return times

def sumarize(times):
    p50 = np.percentile(times, [50])
    p99 = np.percentile(times, [99])
    out = {'p50': p50, 'p99': p99, 'mean': np.mean(times), 'max': np.max(times)}
    return out

def benchmark_batches(predict_fn, input_dim, batch_sizes=[1,8,64], n_runs = 100, warmup=10,seed =0):
    batches = {}
    rng = np.random.default_rng(seed=seed)
    for i in batch_sizes:
        x = rng.random(i,input_dim)
        times = time_fn(predict_fn, n_runs, warmup)
        batches.setdefault(i,sumarize(times))

def compare_fns(fns, n_runs=20, warmup=2):
    final = {}
    for i in fns:
        times = time_fn(fns[i], n_runs, warmup)
        final.setdefault(i,sumarize(times))




