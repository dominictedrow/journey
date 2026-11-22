from collections import deque
import random

class ReplayMemory():

    def __init__(self, maxlen, seed = 0):
        self.memory = deque([], maxlen=maxlen)
        if seed is not None:
            random.seed(seed)

    def append(self, x):
        self.memory.append(x)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)
    
    def __len__(self):
        return len(self.memory)

    
