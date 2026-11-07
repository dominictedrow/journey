import numpy as np, time, timeit,  cProfile
import dis

def f(x, total):
    for i in range(len(x)):
        total += x[i]**2
    return total

#dis.dis(f)

x = np.arange(1_000_000, dtype=np.float32)

def f(x, total):
    for i in range(len(x)):
        total += x[i]**2
    return total

'''
f(x, 0)  # warmup
t = time.perf_counter(); f(x, 0); loop = time.perf_counter() - t
t = time.perf_counter(); np.sum(x**2); npy = time.perf_counter() - t
print(f"loop {loop*1000:.1f} ms, numpy {npy*1000:.2f} ms, ratio {loop/npy:.0f}x")

print(type(x[0]))
print(x[0] is x[0])




for n in [10, 1_000, 100_000, 1_000_000]:
    a = np.ones(n, dtype=np.float32)
    t = timeit.timeit(lambda: np.sum(a), number=1000) / 1000
    print(f"n={n:>9,}  {t*1e6:8.2f} us/call  {t/n*1e9:8.3f} ns/element")


x = np.random.rand(1_000_000).astype(np.float32); lst = x.tolist()
for name, f in [("sum(x)", lambda: sum(x)), ("sum(lst)", lambda: sum(lst)), ("np.sum(x)", lambda: np.sum(x))]:
    print(name, min(timeit.repeat(f, number=1, repeat=5)))


def loop3(x):
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = 3*x[i] + 1 if x[i] > 0.5 else 0
    return out

def loop4(x):
    return np.where(x > 0.5, 3*x + 1, 0)

x = np.random.rand(1_000_000).astype(np.float32)
for name, f in [("loop3(x)", lambda: loop3(x)), ("loop4(lst)", lambda: loop4(x))]:
    print(name, min(timeit.repeat(f, number=1, repeat=5)))

A = np.random.rand(4000, 4000).astype(np.float32)
for name, f in [("Row method", lambda: [A[i, :].sum() for i in range(4000)]), ("Column method", lambda: [A[:, j].sum() for j in range(4000)])]:
    print(name, min(timeit.repeat(f, number=1, repeat=5)))



B = np.random.rand(100000, 10).astype(np.float32)
for name, f in [("Row method", lambda: [np.sum(b) for b in B]), ("Column method", lambda: B.sum(axis=1))]:
    print(name, min(timeit.repeat(f, number=1, repeat=5)))


x = np.random.rand(1_000_000).astype(np.float32)
for name, f in [("Sum: ", lambda: np.sum(x**2)), ("Dot: ", lambda: np.dot(x, x)), ("Einsum: ", lambda: np.einsum('i,i->', x, x) )]:
    print(name, min(timeit.repeat(f, number=1, repeat=5)))
'''


X = np.random.rand(2000, 256).tolist()
W = np.random.rand(256, 256).astype(np.float32)
def normalize(rows):
    out = []
    for r in rows:
        m = sum(r) / len(r)
        out.append([v - m for v in r])
    return np.array(out, dtype=np.float32)


def normal(rows):
    x = np.array(rows)
    out = np.zeros(x.shape)
    m = x.sum(axis=1, keepdims = True) / x.shape[1]
    out = x - m
    return out

    
def pipeline(): return normal(X) @ W
#cProfile.run("pipeline()", sort="cumtime")


B = np.random.rand(100000, 10).astype(np.float32)
for name, f in [("Method 1", lambda: [np.sum(b) for b in B]), ("Method 2", lambda: B.sum(axis=1))]:
    print(name, min(timeit.repeat(f, number=1, repeat=5)))


def normal2(rows):
    x = np.array(rows, dtype=np.float32)
    return x - x.mean(axis=1, keepdims=True)

print("match:", np.allclose(normalize(X), normal2(X), rtol=1e-3, atol=1e-2))
for name, f in [("old pipeline", lambda: normalize(X) @ W), ("new pipeline", lambda: normal2(X) @ W)]:
    print(name, min(timeit.repeat(f, number=1, repeat=5)))

A = np.random.rand(4000, 4000).astype(np.float32)
for name, f in [("rows", lambda: [A[i, :].sum() for i in range(4000)]), ("cols", lambda: [A[:, j].sum() for j in range(4000)])]:
    print(name, min(timeit.repeat(f, number=1, repeat=5)))




