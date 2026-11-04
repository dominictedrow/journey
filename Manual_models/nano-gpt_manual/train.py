import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib as plt
import matplotlib.pyplot as plt
from dataclasses import dataclass


def preprocess(file_path, batch_size,chunk):
    #Load in Data from the filepath
    data = np.memmap(file_path, dtype=np.uint16, mode='r')

    
    ix = [i for i in range(0, len(data), chunk)]
    x = torch.stack([torch.from_numpy(data[i:i+chunk].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+chunk].astype(np.int64)) for i in ix])
    
    x = torch.stack(x[i:i+batch_size] for i in range(0, len(x), batch_size))
    y = torch.stack(y[i:i+batch_size]for i in range(0, len(x), batch_size))

    return x, y

#Config: This is really cool since you can basically define all hyperparameters for any model in one place. Could probably work
#for something else. @dataclass creates the constructor automatically so you can just list the attributes.
@dataclass
class GPTConfig:
    vocab_size: int = 50304
    block_size: int = 256     # context length (max sequence length)
    n_embd: int = 384
    n_head: int = 8
    n_layer: int = 6
    hidden_dim: int = 1024

cfg = GPTConfig()

#Feed Forward Layer:
class Feed(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.layer1 = nn.Linear(cfg.embd, cfg.hidden_dim)
        self.layer2 =  nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.layer3 =  nn.Linear(cfg.hidden_dim, cfg.embd)
    

    def forward(self,x):
        x = self.layer2(self.layer1(x))
        return torch.relu(self.layer3(x))


#MutliHeaded Attention Layer definition:
class AttentionBlock(nn.Module):
    def __init__(self, cfg): 
        super().__init__()
        self.embed_dim1 = cfg.embd
        self.layern1 = nn.LayerNorm(cfg.embd) #Creates a LayerNorm layer
        self.layern2 = nn.LayerNorm(cfg.embd) #Creates a LayerNorm layer
        self.attention = nn.MultiheadAttention(embed_dim=cfg.embd, num_heads=8, batch_first=True)
        self.linear = Feed(cfg.embd, 3*cfg.embd)
        
    

        
    def forward(self, x):  
        y = self.attention(x)
        y = self.layern1(x+y)
        y = self.linear(y)
        y = self.layern2(x+y)
        return y

class Output(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.linear = nn.Linear(cfg.embd)
        self.softmax = nn.Softmax()

    def forward(self,x):
        x = self.softmax(self.linear(x))
        return x


def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.001)
        elif isinstance(m, nn.Embedding):
            nn.init.kaiming_normal(m.weight)
        elif isinstance(m,nn.LayerNorm):
            nn.init.kaiming_normal(m.weights)
        elif isinstance(m, nn.MultiheadAttention):
            nn.init.kaiming_normal(m.weights)

class Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim1) #Creates a token embedding layer
        self.pos_emb = nn.Embedding(cfg.batch_size, cfg.embed_dim1) #Creates a positional embedding layer
        self.out = Output(cfg.embd)
        self.blocks = nn.ModuleList([AttentionBlock for _ in range(32)])


    def foward(self,x):
        x = self.pos_emb(x) + self.token_emb(x)
        for i in self.blocks:
            x = i(x)
        x = self.out(x)
        return x
    



Model.apply(init_weights)



train_input, train_targets = preprocess('train.bin', 32, 8)
val_input, val_targets = preprocess('val.bin', 32, 8)


out_train = Model(train_input)
out_val = Model(val_input)

optimizer = optim.Adam
loss_function = torch.nn.CrossEntropyLoss

optimizer.zero_grad(set_to_none=True)

loss_train = loss_function(out_train, train_targets)
loss_train.backward()

optimizer.step()

loss_val = loss_function(out_val, val_targets)

'''

TO DO LIST:

- Graphing function
- Debug the whole thing
- Add regularization
- Add dropout 
- Add mixed precision training?


'''







