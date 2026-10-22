import numpy as np



def softmax(logits):
    max = np.max(logits, axis=1)
    logits -= max
    logits = np.exp(logits)
    logits = logits/(np.sum(logits, axis=0))
    return logits

def cross(logits, targets):
    p = logits[targets, range(len(targets[:,0]))]
    return np.mean(-1 * np.log(logits[p]))

def perp(loss):
    return np.exp(loss)

def cross_grad(logits, targets):
    # logits: (N, C), targets: (N,) int class ids -> dLoss/dlogits for mean loss
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(z)
    p /= np.sum(p, axis=1, keepdims=True)
    p[np.arange(len(targets)), targets] -= 1
    return p / len(targets)





