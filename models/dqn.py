import torch as torch
import torch.nn as nn


class bloons_bot(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(4272, 16384,bias = True)
        self.layer2 = nn.Linear(16384, 16384, bias = True)
        self.layer3 = nn.Linear(16384, 10499, bias = True)

    def init_params(self):
        nn.init.xavier_normal_(self.layer1)
        nn.init.xavier_normal_(self.layer2)
        nn.init.xavier_normal_(self.layer3)
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)
        nn.init.zeros_(self.layer3.bias)

    def forward(self, x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        x = self.layer3(x)
        return torch.softmax(x) 





