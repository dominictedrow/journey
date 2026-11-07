# Stage 0: NumPy fundamentals, written by hand

I wrote everything in this stage myself, except the backprop in `Module_3_4.py`. The goal was to understand what an ML framework does underneath instead of calling one.

## What's here

| File | What it does |
|---|---|
| `Module_1.py` | Softmax, cross-entropy, perplexity, and the softmax+CE gradient, made numerically stable |
| `Module_2.py` | Matmul written four ways: triple loop, row·column dot, row @ matrix, and vectorized |
| `Module_3_4.py` | A small MLP: init (He for ReLU, Xavier-style for tanh), forward pass, backward pass, minibatch SGD |
| `Module_5.py` | A latency harness: warmup, p50/p95/p99, and throughput across batch sizes |
| `Module_6.py` | Puts it together: benchmarks the matmul variants and trains/benchmarks the MLP on toy blobs |

## What I took away

- **Latency:** Measuring correctly takes more care than it looks: warm up first, report percentiles instead of means, and look at how throughput changes with batch size.
- **Matmul:** Benchmarking the four versions side by side showed how much faster vectorized code is than Python loops, and how quickly the gap grows with matrix size.
- **ML math:** This was the part that mostly didn't work for me. I could write the forward pass, but I didn't get to the point of deriving gradients myself, and the backprop is not my own work. I need to come back to the calculus and linear algebra before later stages lean on it.
