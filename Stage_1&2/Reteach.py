# Reteach 2: count temporaries, skip the conversion, and "fuse" by tiling.
# Run as is, no edits. Answer the questions in chat BEFORE running.
import numpy as np, tracemalloc, timeit

MB = 1e6
n = 10_000_000                      # 10M float32 = 40 MB per array
x, y, w = (np.random.rand(n).astype(np.float32) for _ in range(3))

tracemalloc.start()                 # tracks every array NumPy allocates
def live(): return tracemalloc.get_traced_memory()[0] / MB
def peak(f):
    tracemalloc.reset_peak(); start = tracemalloc.get_traced_memory()[0]
    r = f(); p = tracemalloc.get_traced_memory()[1] - start; del r
    return p / MB

print("A) Counting full-size arrays (each = 40 MB)")
base = live()
t = x * y;   print(f"   t = x * y      -> {live()-base:5.0f} MB live")
z = t + w;   print(f"   z = t + w      -> {live()-base:5.0f} MB live")
del t;       print(f"   del t          -> {live()-base:5.0f} MB live")
del z
print(f"   peak, z = x*y + w (one line)      : {peak(lambda: x*y + w):5.0f} MB")
print(f"   peak, (x*2 + 1)*3 (one line)      : {peak(lambda: (x*2 + 1)*3):5.0f} MB")
def one_buf():
    z = x * y; z += w; return z
print(f"   peak, z = x*y; z += w (one buffer): {peak(one_buf):5.0f} MB")
tracemalloc.stop()

print("\nB) Converting a list vs creating the array directly (2000 x 256)")
L = np.random.rand(2000, 256).tolist()
conv = min(timeit.repeat(lambda: np.array(L, dtype=np.float32), number=10, repeat=3)) / 10
direct = min(timeit.repeat(lambda: np.random.rand(2000, 256).astype(np.float32), number=10, repeat=3)) / 10
print(f"   np.array(list) {conv*1000:6.1f} ms   created as array {direct*1000:6.1f} ms   ratio {conv/direct:4.1f}x")

print("\nC) Softmax on a 4096 x 4096 'scores' matrix (64 MB)")
S = np.random.rand(4096, 4096).astype(np.float32)
def whole():                        # each step = a full 64 MB pass through RAM
    a = S * 0.125
    e = np.exp(a - a.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)
def tiled(rows=32):                 # 32 rows = 512 KB tile, stays in cache for all 4 steps
    out = np.empty_like(S)
    for i in range(0, 4096, rows):
        o = out[i:i+rows]
        np.multiply(S[i:i+rows], 0.125, out=o)
        o -= o.max(axis=1, keepdims=True)
        np.exp(o, out=o)
        o /= o.sum(axis=1, keepdims=True)
    return out
tw = min(timeit.repeat(whole, number=3, repeat=5)) / 3
tt = min(timeit.repeat(tiled, number=3, repeat=5)) / 3
print(f"   whole array {tw*1000:6.1f} ms   tiled {tt*1000:6.1f} ms   ratio {tw/tt:4.2f}x   "
    f"match {np.allclose(whole(), tiled())}")
















































'''
# Reteach: temporary arrays vs per-call overhead. Just run it, no edits needed.
import numpy as np, timeit

def us(f, k=20):  # best-of-5 microseconds per call
    return min(timeit.repeat(f, number=k, repeat=5)) / k * 1e6

print("A) np.sum(x**2) vs np.dot(x, x)")
for n in [4, 1_000_000]:
    x = np.random.rand(n).astype(np.float32)
    s, d = us(lambda: np.sum(x**2), 200), us(lambda: np.dot(x, x), 200)
    print(f"   n={n:>10,}  sum(x**2) {s:9.2f} us   dot {d:9.2f} us   "
        f"temp array = {x.nbytes:>10,} bytes")

print("\nB) (x*2 + 1)*3  [2 temporaries]  vs  y=x*2; y+=1; y*=3  [1 buffer]")
for n in [1_000, 1_000_000, 10_000_000]:
    x = np.random.rand(n).astype(np.float32)
    def chained(): return (x * 2 + 1) * 3
    def one_buffer():
        y = x * 2; y += 1; y *= 3; return y
    c, o = us(chained), us(one_buffer)
    print(f"   n={n:>10,} ({x.nbytes/1e6:6.3f} MB each)  chained {c:10.1f} us   "
        f"one buffer {o:10.1f} us   ratio {c/o:4.2f}x   match {np.allclose(chained(), one_buffer())}")

print("\nC) in-place on x itself")
x = np.array([1., 2., 3.], dtype=np.float32); original = x.copy()
x *= 2; x += 1; x *= 3
print(f"   x before {original}   x after {x}")

#Reteach drills: temporaries. Run as is. Write your answers to the 3 questions FIRST.
import numpy as np, timeit

def us(f, k=10):
    return min(timeit.repeat(f, number=k, repeat=5)) / k * 1e6

# Drill 1: n = 1,000
x = np.random.rand(1_000).astype(np.float32)
s, d = us(lambda: np.sum(x**2), 2000), us(lambda: np.dot(x, x), 2000)
print(f"D1  n=1,000   sum(x**2) {s:7.2f} us   dot {d:7.2f} us   ratio {s/d:4.2f}x")

# Drill 2 (softmax): softmax on 10M floats (the core of attention)
x = np.random.rand(10_000_000).astype(np.float32)
def softmax_naive():
    ex = np.exp(x - x.max())
    return ex / ex.sum()
def softmax_one_buf():
    y = x - x.max()
    np.exp(y, out=y)
    y /= y.sum()
    return y
t1, t2 = us(softmax_naive), us(softmax_one_buf)
print(f"D2  softmax   naive {t1/1000:7.1f} ms   one buffer {t2/1000:7.1f} ms   "
    f"ratio {t1/t2:4.2f}x   match {np.allclose(softmax_naive(), softmax_one_buf())}")
    '''