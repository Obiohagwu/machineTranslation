
from logging import root
from posixpath import split
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
import math


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# Now we have to construct a model based on transformer architecture


# implementing token "embedder"
class Embed(nn.Module):
    def __init__(self, vocab_size, model_dim):
        super(Embed, self).__init__()
        self.embed = nn.Embed(vocab_size, model_dim)

    def forward(self, x):
        return self.embed(x)


# implementing of positional encoder for permutation variance
class PositionalEncoder(nn.Module):
    def __init__(self, model_dim, maximum_seq_len=80):
        super().__init__()
        self.model_dim = model_dim

        # create constant 'pe' (position encoding) matrix with values dependant on 
        # pos (position) and i (index)
        pe = torch.zeros(max_seq_len, d_model)
        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
                
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # make embeddings fairly large
        x*=math.sqrt(self.model_dim)
        # add a constant
        sequence_len = x.size(1)
        x+=torch.autograd.Variable(self.pe[:,:sequence_len], requires_grad=False).to(device=DEVICE)
        return x


# Let's implement multihead attn
class multiHeadAttention(nn.Module):
    def __init__(self, heads, model_dim, dropout = 0.1):
        super().__init__()

        self.model_dim = model_dim
        self.d_k = model_dim // heads # key dim
        self.h = heads
        self.q_linear = nn. Linear(model_dim, model_dim) # Query
        self.v_linear = nn. Linear(model_dim, model_dim) # Value
        self.k_linear = nn. Linear(model_dim, model_dim) # Key
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(model_dim, model_dim)

    def forward(self, q, k ,v , mask=None):
        bs = q.size(0)

        # Given our understanding of multihead attention mechanisms
        # we will now perform linear operations and split qkv into h heads
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # we transpose to transform dimension intio bs*h*s1*model_dim

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attn score
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(bs, -1, self.model_dim)

        output = self.out(concat)
        return output


# We have to define the attention function
def attention(q,k,v, d_k, mask=None, dropout=None):
    # This function can be found in attn paper
    scores = torch.matmul(q, k.transpose(-2, -2)).math.sqrt(d_k)

    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask==0,-1e9)
    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output












def test():
    print("Just checking for tuntime errs!")

if __name__=="__main__":
    test()
    