import torch as torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml


boat = (fetch_openml(name = 'titanic', version = 1))
x = pd.DataFrame(boat.data)
y = pd.DataFrame(boat.target)
x.drop(columns=['name', 'home.dest', ], inplace = True)

print(x['cabin'].value_counts())
print(x['ticket'].value_counts())
print(x['embarked'].value_counts())
print(x['boat'].value_counts())


def preprocess():
    boat = (fetch_openml(name = 'titanic', version = 1))
    x = pd.DataFrame(boat.data)
    y = pd.DataFrame(boat.target)
    x.drop(columns=['name', 'home.dest', ], inplace = True)
    return x, y

class titanic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10,2, bias = True)
    def init_params(self):
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.xavier_normal_(self.layer1.bias)
    def forward(self,x):
        z = self.layer1(x)
        z = torch.sigmoid(z)
        return z




