import numpy as np
import math

def first(a,b):
    final = []
    for i in range(len(b[0,:])):
        for x in range(len(a[:,0])):
            insert = []
            added = 0
            for t in range(len(b[0,:])):
                added += a[x,t] * b[t,i]
            insert.append(added)
        final.append(insert)
    return final

def second(a,b):
    final = np.zeros(a.shape[1], b.shape[0])
    for i in range(len(b[0,:])):
        for x in range(len(a[:,0])):
            final[i] = np.dot(a[x], b[i])
    return final

def third(a,b):
    final = []
    for i in range(len(a[:,0])):
        final[:,i] = a[i] @ b
    return final



def fourth(a,b):
    return a @ b




