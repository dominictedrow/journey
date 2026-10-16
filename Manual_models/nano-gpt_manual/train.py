import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib as plt
import matplotlib.pyplot as plt



def preprocess(file_path, batch_size,chunk):
    #Load in Data from the filepath
    data = np.memmap(file_path, dtype=np.uint16, mode='r')

    
    ix = [i for i in range(0, len(data), chunk)]
    x = torch.stack([torch.from_numpy(data[i:i+chunk].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i+1:i+1+chunk].astype(np.int64)) for i in ix])
    
    x = torch.stack(x[i:i+batch_size] for i in range(0, len(x), batch_size))
    y = torch.stack(y[i:i+batch_size]for i in range(0, len(x), batch_size))

    return x, y

#Feed Forward Layer:
class Feed(nn.Module):
    def __init__(self, shape, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(shape, hidden_dim)
        self.layer2 =  nn.Linear(hidden_dim, hidden_dim)
        self.layer3 =  nn.Linear(hidden_dim, shape)
    
    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.001)

    def forward(self,x):
        x = self.layer2(self.layer1(x))
        return torch.relu(self.layer3(x))


#MutliHeaded Attention Layer definition:
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim1): 
        super().__init__()
        self.embed_dim1 = embed_dim1
        self.layern1 = nn.LayerNorm(embed_dim1) #Creates a LayerNorm layer
        self.layern2 = nn.LayerNorm(embed_dim1) #Creates a LayerNorm layer
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim1, num_heads=8, batch_first=True)
        self.linear = Feed(embed_dim1, 3*embed_dim1)
        self.apply(self.init_weights) #initializes weights
    
    def init_weights(self, m):
        if isinstance(m,nn.LayerNorm):
            nn.init.kaiming_normal(m.weights)
        elif isinstance(m, nn.MultiheadAttention):
            nn.init.kaiming_normal(m.weights)
        
    def forward(self, x,embed_dim1):  
        x = self.attention(x)
        x = self.linear(self.layern1(x))
        x = self.layern2(x)
        x = x[:,embed_dim1]
        return x

class Output(nn.Module):
    def __init__(self,embed_dim1):
        super().__init__()
        self.linear = nn.Linear(embed_dim1)
        self.softmax = nn.Softmax()

    def init_weights(self,m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.001)
    
    def forward(self,x):
        x = self.softmax(self.linear(x))
        return x




class Model(nn.Module):
    def __init__(self,  vocab_size, batch_size, embed_dim1, chunk):
        self.token_emb = nn.Embedding(vocab_size, embed_dim1) #Creates a token embedding layer
        self.pos_emb = nn.Embedding(batch_size, embed_dim1) #Creates a positional embedding layer
        self.out = Output(embed_dim1)
        self.blocks = nn.ModuleList([AttentionBlock for _ in range(32)])

    def init_weights(self,m):
        if isinstance(m, nn.Embedding):
        # Xavier/Glorot initialization for Linear layers (ideal for Tanh/Sigmoid)
            nn.init.kaiming_normal(m.weight)

    def foward(self,x):
        x = self.pos_emb(x) + self.token_emb(x)
        for i in self.blocks:
            x = i(x)
        x = self.out(x)
        return x
    


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
- Add a weight init function
- Add residual connections
- Add a better intializer
- Add mixed precision training?


'''







