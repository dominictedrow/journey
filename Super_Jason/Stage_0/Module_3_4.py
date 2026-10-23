import numpy as np
import math

def init_params(sizes, seed = 0):
    rng = np.random.default_rng(seed)
    params= []
    for i in sizes:
        params.append((rng.random(i), rng.random()))
    return params

def forward(params, X):
    cache = [X]
    for i in range(len(params)-1):
        X = X @ np.array(params[i][0]) + np.array(params[i][1])
        cache.append(X)
        X = np.maximum(0,X)
    logits = X @ np.array(params[-1][0]) + np.array(params[-1][1])
    return logits, cache

def cross_grad(logits, targets):
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(z)
    p /= np.sum(p, axis=1, keepdims=True)
    p[np.arange(len(targets)), targets] -= 1
    return p / len(targets)


def backward(params, dlogits, cache):
    grads = []
    dZ = dlogits
    for i in reversed(range(len(params))):
        W, b = np.array(params[i][0]), params[i][1]
        A = cache[0] if i == 0 else np.maximum(0, cache[i])
        db = dZ.sum(axis=0) if np.ndim(b) else dZ.sum()
        grads.append((A.T @ dZ, db))
        if i > 0:
            dZ = (dZ @ W.T) * (cache[i] > 0)
    return grads[::-1]

def predict(params,X):
    logits, _ = forward(params, X)
    out = np.argmax(logits, axis=1)
    return out

def train(X, y, params, lr=0.1, epochs =50, batch_size=64, seed = 0,):
    rng = np.random.default_rng(seed)
    history = []
    for i in range(epochs):
        X = rng.shuffle(X)
        batched = [X[i] for i in range(0, len(X)-batch_size, batch_size)]
        loss = []
        for x in batched:
            logits, cache = forward(params,x)
            dlogits = cross_grad(logits, cache)
            loss.append(-np.sum(y*np.log(logits)))
            dw, db = backward(params, dlogits, cache)
            for i in params:
                i[0] = -dw * i[0] * lr
                i[1] = -db * i[1] * lr
        history.append(np.mean(loss))
    return params, history


            




