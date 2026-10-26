import numpy as np



def softmax(logits):
    maxx = np.max(logits, axis=1, keepdims=True)
    z = logits - maxx
    e = np.exp(z)
    return e/(np.sum(e, axis=1,keepdims=True))

def cross_entropy(logits, targets):
    maxx = np.max(logits, axis=1, keepdims=True)
    z = logits - maxx
    e = z - np.log(np.sum(np.exp(z), axis=1, keepdims=True))
    return  -np.mean(e[np.arange(len(targets)), targets])

def perplexity(loss):
    return np.exp(loss)

def softmax_ce_grad(logits, targets):
    # logits: (N, C), targets: (N,) int class ids -> dLoss/dlogits for mean loss
    z = logits - np.max(logits, axis=1, keepdims=True)
    p = np.exp(z)
    p /= np.sum(p, axis=1, keepdims=True)
    p[np.arange(len(targets)), targets] -= 1
    return p / len(targets)





