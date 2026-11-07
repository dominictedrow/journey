import numpy as np
from Module_1 import cross_entropy, softmax_ce_grad
import math

def relu(inputs):
    return np.maximum(0,inputs)

def tanh(inputs):
    return np.tanh(inputs)

def init_params(sizes, seed = 0, activation = "relu"):
    rng = np.random.default_rng(seed)
    params= []
    for i in range(len(sizes)-1):
        if activation == "relu":    
            params.append((rng.normal(0, np.sqrt(2 / sizes[i]), (sizes[i], sizes[i+1])), np.zeros(sizes[i+1])))
        elif activation == "tanh":
            params.append((rng.normal(0,np.sqrt(1/sizes[i]),(sizes[i], sizes[i+1])), np.zeros(sizes[i+1])))
    return params

def forward(params, X, activation = "relu"):
    cache = [X]
    for i in range(len(params)-1):
        X = X @ np.array(params[i][0]) + np.array(params[i][1])
        if activation == "relu":
            cache.append(X)
            X = relu(X)
        elif activation == "tanh":
            cache.append(X)
            X = tanh(X)
            cache.append(X)
    logits = X @ np.array(params[-1][0]) + np.array(params[-1][1])
    return logits, cache



def backward(params, cache, dlogits, activation = "relu"):
    grads = []
    dZ = dlogits
    for i in reversed(range(len(params))):
        W, b = np.array(params[i][0]), params[i][1]
        if i == 0:
            A = cache[0]
        elif activation == "tanh":
            A = cache[2*i]  # tanh cache: [X, Z1, A1, Z2, A2, ...]
        else:
            A = relu(cache[i])  # relu cache: [X, Z1, Z2, ...]
        db = dZ.sum(axis=0) if np.ndim(b) else dZ.sum()
        grads.append((A.T @ dZ, db))
        if i > 0:
            if activation == "tanh":
                dZ = (dZ @ W.T) * (1 - A**2)
            else:
                dZ = (dZ @ W.T) * (cache[i] > 0)
    return grads[::-1]

def predict(params,X, activation="relu"):
    logits, _ = forward(params, X,activation)
    out = np.argmax(logits, axis=1)
    return out

def train(X, y, sizes, lr=0.1, epochs =50, batch_size=64, seed = 0, activation = "relu"):
    params = init_params(sizes,seed,activation)
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
            logits, cache = forward(params,inputs[x], activation)
            dlogits = softmax_ce_grad(logits, targets[x])
            loss.append(cross_entropy(logits, targets[x]))
            grads= backward(params, cache, dlogits, activation)
            param1 = []
            for n in range(len(params)):
                W, b = params[n]
                dW, db = grads[n]
                param1.append(((W - lr * dW), (b - lr * db)))
            params = param1
        history.append(np.mean(loss))
    return params, history







