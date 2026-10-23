from Module_2 import first, second, third, fourth 
from Module_5 import time_fn, sumarize, benchmark_batches, compare_fns
from Module_3_4 import predict, train
import numpy as np

rng = np.random.default_rng()
A = rng.random(64,64)
B = rng.random(64,64)
C = rng.random(16,16)
D = rng.random(16,16)
E = rng.random(128,128)
F = rng.random(128,128)
fns1 = {'first':first(A, B),'second': second(A,B),'third': third(A,B), 'fourth': fourth(A,B)}
fns2 = {'first':first(C, D),'second': second(C,D),'third': third(C,D), 'fourth': fourth(C,D)}
fns3 = {'first':first(E, F),'second': second(E,F),'third': third(E,F), 'fourth': fourth(E,F)}

comp1 = compare_fns(fns1)
comp2 = compare_fns(fns2)
comp3 = compare_fns(fns3)






