import torch as torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris




def preprocess():
    iris = load_iris()
    x = torch.tensor(iris.data)
    y = torch.tensor(iris.target)

    indices = torch.randperm(x.shape[0])
    x = x[indices]
    y = y[indices]
    
    x_train = x[:110]
    y_train = y[:110]
    x_test = x[110:]
    y_test = y[110:]

    m, v = x_train.mean(dim=0), x_train.std(dim=0)
    #No need to batch only 150 samples
    x_train = (x_train - m)/ v
    x_test = (x_test - m) / v
    y_train = torch.nn.functional.one_hot(y_train, num_classes=3)# torch I guess just has a one hot encoding function
    y_test = torch.nn.functional.one_hot(y_test, num_classes=3)

    x_train = x_train.to(torch.float32)
    x_test = x_test.to(torch.float32)
    y_train = y_train.to(torch.float32)
    y_test = y_test.to(torch.float32)
    
    return x_train, y_train, x_test, y_test
    



class IrisClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(4, 3, bias = True)

    def init_params(self):
        nn.init.kaiming_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
    
    def forward(self, x):
        x = self.linear1(x)
        x = torch.softmax(x, dim=1)
        return x


#Initialize everything:        
model = IrisClassifier()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
func = torch.nn.MSELoss()
x_train, y_train, x_test, y_test = preprocess()

def train(model, optimizer, func, x_train, y_train):
    model.init_params()
    for epoch in range(1000):
        optimizer.zero_grad()
        y_pred = model(x_train)
        loss = func(y_pred, y_train)
        loss.backward()
        optimizer.step()
    return loss.item()

def test(model, func, x_test, y_test):
    with torch.no_grad():
        y_pred = model(x_test)
        loss = func(y_pred, y_test)
    return loss.item()

loss = train(model, optimizer, func, x_train, y_train)
test_loss = test(model, func, x_test, y_test)
print(str(loss) + " " +str(test_loss))





