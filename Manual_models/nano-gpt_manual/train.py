import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib as plt
import matplotlib.pyplot as plt



def preprocess(file_path, batch_size):
    #Load in Data from the filepath
    data = np.memmap(file_path, dtype=np.uint16, mode='r')
    
    #Batch function and data chunk function
    def batch(data, batch_size,chunk):
        chunk += 1
        n = len(data)
        chunks = []
        for i in range(0, n, chunk):
            chunks.append(data[i:i + chunk])
        y = [i[1:] for i in chunks]
        x = []
        for i in chunks:
            chunk1 =[]
            for j in range(chunk):
                chunk1.append(i[0:j])
            x.append(chunk1)
        x = torch.tensor(x)
        y = torch.tensor(y)
        x_batch= [torch.tensor(x[:,i]) for i in range(0,len(chunks),batch_size)]
        y_batch= [torch.tensor(y[:,i]) for i in range(0,len(chunks),batch_size)]
        return x_batch, y_batch 

    #Call the batch function to create batches of data
    data = batch(data, batch_size)
    return data


#Feed Forward Layer:
class Feed(nn.module):
    def __init__(self, shape, hidden_dim):
        super().__init__()
        self.layer1 = nn.Linear(shape, hidden_dim)
        self.layer2 =  nn.Linear(hidden_dim, shape)
    def forward(self,input):
        x = torch.relu(self.layer1(input))
        return self.layer2(x)


#MutliHeaded Attention Layer definition:
class AttentionBlock(nn.Module):
    def __init__(self, embed_dim1, chunk): 
        super().__init__()
        self.embed_dim1 = embed_dim1
        self.layern1 = nn.LayerNorm(embed_dim1) #Creates a LayerNorm layer
        self.layern2 = nn.LayerNorm(embed_dim1) #Creates a LayerNorm layer
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim1, num_heads=chunk, batch_first=True)
        self.linear = Feed(embed_dim1, 3*embed_dim1)
        self.apply(self.init_weights) #initializes weights
    

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
    


class Model(nn.Module):
    def __init__(self,  vocab_size, batch_size, embed_dim1, chunk):
        self.token_emb = nn.Embedding(vocab_size, embed_dim1) #Creates a token embedding layer
        self.pos_emb = nn.Embedding(batch_size, embed_dim1) #Creates a positional embedding layer
        self.out = Output(embed_dim1)
        self.blocks = nn.ModuleList([AttentionBlock for _ in range(32)])

    def foward(self,x):
        x = self.pos_emb(x) + self.token_emb(x)

        for i in self.blocks:
            x = i(x)
        x = self.out(x)

        return x
    


train_input, train_targets = preprocess('C:\Users\blazi\AI_projects\Manual_models\nano-gpt_manual\train.bin', 32)
val_input, val_targets = preprocess('C:\Users\blazi\AI_projects\Manual_models\nano-gpt_manual\val.bin', 32)


out_train = Model(train_input)
out_val = Model(val_input)


loss_function = torch.nn.CrossEntropyLoss

loss_train = loss_function(out_train, train_targets)
loss_train.backward()
loss_val = loss_function(out_val, val_targets)









