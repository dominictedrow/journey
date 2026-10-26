from Module_1 import perplexity
from Module_2 import matmul_loops, matmul_rowdot, matmul_rows, matmul_vec 
from Module_5 import time_fn, summarize, benchmark_batches, compare_fns
from Module_3_4 import predict, train, init_params
import numpy as np

rng = np.random.default_rng()
A = rng.random((64,64))
B = rng.random((64,64))
C = rng.random((16,16))
D = rng.random((16,16))
E = rng.random((128,128))
F = rng.random((128,128))
fns1 = {'first':lambda: matmul_loops(A, B),'second': lambda: matmul_rowdot(A,B),'third': lambda:matmul_rows(A,B), 'fourth': lambda: matmul_vec(A,B)}
fns2 = {'first':lambda: matmul_loops(C, D),'second': lambda: matmul_rowdot(C,D),'third': lambda:matmul_rows(C,D), 'fourth': lambda: matmul_vec(C,D)}
fns3 = {'first':lambda: matmul_loops(E, F),'second': lambda: matmul_rowdot(E,F),'third': lambda:matmul_rows(E,F), 'fourth': lambda: matmul_vec(E,F)}

comp1 = compare_fns(fns1)
comp2 = compare_fns(fns2)
comp3 = compare_fns(fns3, n_runs=3, warmup=1)

print(comp1)
print(comp2)
print(comp3)




def blobs(n_per=100, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 4], [4, -2], [-4, -2]], dtype=float)
    X = np.vstack([c + rng.normal(size=(n_per, 2)) for c in centers])
    y = np.repeat(np.arange(3), n_per)
    return X, y

X, y = blobs()
sizes = [2, 32, 3]

params, history = train(X, y, sizes, lr=0.1, epochs=40, batch_size=32, seed=0)
print("loss first/last:", history[0], history[-1])
print("perplexity first/last:", perplexity(history[0]), perplexity(history[-1]))
print("accuracy:", (predict(params, X) == y).mean())

print(perplexity(history[-1]))
print((predict(params, X) == y).mean())
batch_times = benchmark_batches(lambda X: predict(params, X), input_dim=2)

for b, i in batch_times.items():
    print(str(i['p50']), str(i['p95']), str(i['p99']), str(i['mean']), str(i['throughput']))




