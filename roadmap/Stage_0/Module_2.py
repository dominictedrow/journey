import numpy as np
import math

def matmul_loops(a,b):
    final = []
    if a.shape[1] == b.shape[0]:   
        for i in range(a.shape[0]):
            row = []
            for x in range(b.shape[1]):
                added = 0
                for t in range(a.shape[1]):
                    added += a[i,t] * b[t,x]
                row.append(added)
            final.append(row)
        return np.array(final)
    else:
        raise ValueError('Shapes not matched')

def matmul_rowdot(a,b):
    final = np.zeros((a.shape[0], b.shape[1]))
    for i in range(len(b[0,:])):
        for x in range(len(a[:,0])):
            final[x,i] = np.dot(a[x], b[:,i])
    return final

def matmul_rows(a,b):
    final =np.zeros((a.shape[0], b.shape[1]))
    for i in range(len(a[:,0])):
        final[i] = a[i] @ b
    return final



def matmul_vec(a,b):
    return a @ b




