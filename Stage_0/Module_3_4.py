import numpy as np
from Module_1 import cross_entropy, softmax_ce_grad
import math

def init_params(sizes, seed = 0):
    rng = np.random.default_rng(seed)
    params= []
    for i in range(len(sizes)-1):
        params.append((rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i+1])), np.zeros(sizes[i+1])))
    return params

def forward(params, X):
    cache = [X]
    for i in range(len(params)-1):
        X = X @ np.array(params[i][0]) + np.array(params[i][1])
        cache.append(X)
        X = np.maximum(0,X)
    logits = X @ np.array(params[-1][0]) + np.array(params[-1][1])
    return logits, cache



def backward(params, cache, dlogits,):
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

def train(X, y, sizes, lr=0.1, epochs =50, batch_size=64, seed = 0,):
    params = init_params(sizes,seed)
    rng = np.random.default_rng(seed)
    history = []
    for i in range(epochs):
        random = rng.permutation(len(X))
        X = X[random]
        y = y[random]
        inputs= [X[i:i+batch_size] for i in range(0, len(X), batch_size)]
        targets = [y[i:i+batch_size] for i in range(0, len(y), batch_size)]
        loss = []
        for x in range(len(inputs)):
            logits, cache = forward(params,inputs[x])
            dlogits = softmax_ce_grad(logits, targets[x])
            loss.append(cross_entropy(logits, targets[x]))
            grads= backward(params, cache, dlogits)
            param1 = []
            for n in range(len(params)):
                W, b = params[n]
                dW, db = grads[n]
                param1.append(((W - lr * dW), (b - lr * db)))
            params = param1
        history.append(np.mean(loss))
    return params, history







