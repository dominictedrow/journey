import numpy as np
from dataclasses import dataclass

x = np.fromfile('C:\Users\blazi\Roadmap\Stage_1&2\data\mnist\train-images-idx3-ubyte', dtype=np.uint8)
y = np.fromfile('C:\Users\blazi\Roadmap\Stage_1&2\data\mnist\train-labels-idx1-ubyte', dtype=np.uint8)

x_test = np.fromfile('C:\Users\blazi\Roadmap\Stage_1&2\data\mnist\t10k-images-idx3-ubyte', dtype=np.uint8)
y_test = np.fromfile('C:\Users\blazi\Roadmap\Stage_1&2\data\mnist\t10k-labels-idx1-ubyte', dtype=np.uint8)


rng = np.default_rng()
W1 = rng.normal(0, 0.01, size=(784, 256))
W2 = rng.normal(0, 0.01, size=(256, 256))
W3 = rng.normal(0, 0.01, size=(256, 10))

B1 = rng.normal(0, 0.01, size = (256))
B2 = rng.normal(0, 0.01, size = (256))


@dataclass
class config:
    batch_size: int = 64
    epochs: int = 1000
    lr: float = 0.001


def preprocess(x,y):
    x = x.reshape(-1, 784)
    y = y.reshape(-1, 784)
    x = x/255
    y = y/255
    return x, y


def batch(x,y, batch_size):
    indexes = np.random.randint(0,len(x[0], 16))
    x_batch = np.hstack([x[i:i+batch_size] for i in indexes])
    y_batch = np.hstack([y[i:i+batch_size] for i in indexes])
    return x_batch, y_batch


def softmax(x):
    x = np.exp(x)/(np.sum(np.exp(x), axis = 0))




def forward(x,y,W1,W2,W3,B1,B2):
    x,y = preprocess(x,y)
    x, y =  batch(x,y)
    
    z = x @ W1 + B1
    #Batch norm
    z = np.tanh(z)
    
    z = x @ W2 + B2
    #Batch norm
    z = np.tanh(z)

    z = x @ W3
    #Batch norm
    z = np.tanh(z)
    z = softmax(z)

    loss = cross(z,y)
    return z, loss


















