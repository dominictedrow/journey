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
    x.drop(columns=['name', 'home.dest','boat', 'body'], inplace = True)

#Make lookup tables and apply to user features
    for col in ['ticket', 'embarked', 'sex', 'cabin']:
        x[col] = pd.factorize(x[col])[0]
    
    #Shuffle
    shuffled_t = x.sample(frac=1, random_state=42).index
    x = x.loc[shuffled_t].reset_index(drop=True)
    y = y.loc[shuffled_t].reset_index(drop=True)
    
    #Split Data
    x_train = x.iloc[:1000]
    y_train = y.iloc[:1000]
    y_test = y.iloc[1000:]
    x_test = x.iloc[1000:]
    
    #Fill NaN Values
    x_train = x_train.fillna(0)
    x_test = x_test.fillna(0)

    #Convert to Tensors
    x_train = torch.tensor(x_train.to_numpy().astype(np.float32), dtype=torch.float32)
    x_test = torch.tensor(x_test.to_numpy().astype(np.float32), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy().astype(np.float32), dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy().astype(np.float32), dtype=torch.float32)

    #Normalize values
    m, v = x_train.mean(dim=0), x_train.std(dim=0)
    x_train = (x_train - m)/v
    x_test = (x_test -m)/v
    
    #Batch
    train_r = x_train.shape[0] % 64
    test_r = x_test.shape[0] % 64
    
    x_train = torch.stack([x_train[i:i+64] for i in range(0, (x_train.shape[0]-train_r), 64)])
    x_test = torch.stack([x_test[i:i+64] for i in range(0, (x_test.shape[0]-test_r), 64)])
    y_train = torch.stack([y_train[i:i+64] for i in range(0, (y_train.shape[0]-train_r), 64)])
    y_test = torch.stack([y_test[i:i+64] for i in range(0,(y_test.shape[0]-test_r), 64)])
    
    
    
    return x_train, y_train, x_test, y_test, 


class titanic(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(9,1, bias = True)
    def init_params(self):
        nn.init.xavier_normal_(self.layer1.weight)
        nn.init.zeros_(self.layer1.bias)
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
        for x, y in zip(x_train,y_train):
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()
    #Evaluate the model
    with torch.no_grad():
        y_pred_test = model(x_test)
        loss_test = criterion(y_pred_test, y_test)
        print(f"Test Loss: {loss_test.item()}")

train()



