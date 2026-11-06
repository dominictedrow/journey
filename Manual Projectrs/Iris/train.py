import torch as torch
import torch.nn as nn
import pandas as pd
import matplotlib as plt
from sklearn.datasets import load_iris

iris = load_iris()
df = pd.DataFrame(data = iris.data, columns= iris.feature_names)

df['species']= iris.target
print(df)




class IrisClassifier(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 3, bias = True)

    def init_params(self):
        nn.init.kaiming_normal_(self.linear1.weights)
        nn.init.zeros_(self.linear1.bias)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.softmax(x)
        return x


        
model = IrisClassifier
optimzer = torch.optim.Adam(model.parameters, lr = 0.001)
func = torch.nn.MSELoss()


def train(m,x,y):





