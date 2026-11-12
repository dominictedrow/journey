import torch as torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml


def preprocess():
    #Get data
    boat = (fetch_openml(name = 'titanic', version = 1))
    x = pd.DataFrame(boat.data)
    y = pd.DataFrame(boat.target)
    x.drop(columns=['name', 'home.dest', ], inplace = True)

    #Split Data
    x_train = x.iloc[:1000]
    y_train = y.iloc[:1000]
    y_test = y.iloc[1000:]
    x_test = x.iloc[1000:]
    
    #Make lookup tables and apply to user features
    unique_t = x['ticket'].unique
    table_t = torch.vstack(unique_t, torch.arrange(len(unique_t)))
    for i in range(table_t.shape[1]):
        filter = x['ticket'] == unique_t[i]
        x['ticket'][filter] = table_t[1][i]
    
    unique_e = x['embarked'].unique
    table_e = torch.vstack(unique_e, torch.arrange(len(unique_e)))
    for i in range(table_e.shape[1]):
            filter = x['embarked'] == unique_e[i]
            x['embarked'][filter] = table_e[1][i]
    
    unique_c = x['cabin'].unique
    table_c = torch.vstack(unique_c, torch.arrange(len(unique_c)))
    for i in range(table_c.shape[1]):
                filter = x['cabin'] == unique_c[i]
                x['cabin'][filter] = table_c[1][i]

    unique_b = x['boat'].unique
    table_b = torch.vstack(unique_b, torch.arrange(len(unique_b)))
    for i in range(table_b.shape[1]):
                filter = x['boat'] == unique_b[i]
                x['boat'][filter] = table_b[1][i]
    
    x_train = x_train.fillna(0)
    x_test = x_test.fillna(0)
    
    x_train = torch.tensor(x_train.to_numpy(), dtype=torch.float32)
    x_test = torch.tensor(x_test.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32)
    
    #Normalize values
    m, v = x.mean(dim=0), x.std(dim=0)
    x_train = (x_train - m)/v
    x_test = (x_test -m)/v
    
    return x_train, y_train, x_test, y_test, 


'''
Need a plan for if the thing is NaN
'''

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


model = titanic()
def train():
    x_train, y_train, x_test, y_test = preprocess()
    model.init_params()
    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for epoch in range(100):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
    #Evaluate the model
    with torch.no_grad():
        y_pred_test = model(x_test)
        loss_test = criterion(y_pred_test, y_test)
        print(f"Test Loss: {loss_test.item()}")

train()



