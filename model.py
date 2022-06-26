
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



# We implement an encoder and decoder layer class that will be used in encode and decode class

# After implementing those, we will finally get to final transformer mechanism

#implement norm class
class Norm(nn.Module):
    def __init__(self, model_dim, eps=1e-6):
        super().__init__()

        self.size=model_dim

        # We implemet 2 learnable paramseters (alpha, bias) to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.ones(self.size))
        self.eps = eps 

    def forward(self, x):
        # Refer to paper to see norm formula/definition
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

# We implement an encoder layer with a feedforward layer and a a mutliheadattn layer
class EncoderLayer(nn.Module):
    # Refer to papers explicating the encoder layer for understanding
    def __init__(self, model_dim, heads, dropout=0.1):
        super().__init__()
        #initializing two norm params we defines earlier 
        self.norm1 = Norm(model_dim)
        self.norm2 = Norm(model_dim)
        self.attntn = multiHeadAttention(heads, model_dim)
        self.feed_forward = FeedForward(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x_int = self.norm1(x)
        x+=self.dropout1(self.attntn(x_int, x_int, x_int, mask))
        x_int = self.norm2(x)
        x+=self.dropout2(self.feed_forward(x_int))
        return x 


# We implement an encoder layer with a feedforward layer and 2 mutliheadattn layer       

class DecodeLayer(nn.Module):
    def __init__(self, model_dim, heads, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(model_dim)
        self.norm2 = Norm(model_dim)
        self.norm3 = Norm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.attntn1 = multiHeadAttention(heads, model_dim)
        self.attntn2 = multiHeadAttention(heads, model_dim)
        self.feed_forward = FeedForward(model_dim).to(device=DEVICE)

        def forward(self, x, exp_out, source_mask, target_mask):
            x_int = self.norm1(x)
            x+=self.dropout1(self.attntn1(x_int, x_int, x_int, target_mask))
            x_int = self.norm2(x)
            x+=self.dropout2(self.attntn2(x_int, exp_out, exp_out, source_mask))
            x_int = self.norm3(x)
            x+=self.dropout3(self.feed_forward(x_int))
            return x 

# We can then build a convenient cloning function that can generate multiple layers:
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# We can finally implement the encoder and decoder modules

class Encoder(nn.Module):
    def __init__(self, vocab_size, model_dim, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embed(vocab_size, model_dim)
        self.pos_enc = PositionalEncoder(model_dim)
        self.layers = get_clones(EncoderLayer(model_dim, heads), N)
        self.norm=Norm(model_dim)

    def forward(self, source, mask):
        x = self.embed(source)
        x = self.pos_enc(x)
        for j in range(N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    def __init__(self, vocab_size, model_dim, N, heads):
        super().__init__()
        self.N = N
        self.embed = Embed(vocab_size, model_dim)
        self.pos_enc = PositionalEncoder(model_dim)
        self.layers = get_clones(EncoderLayer(model_dim, heads), N)
        self.norm=Norm(model_dim)

    def forward(self, target, exp_out, source_mask, target_mask):
        x = self.embed(target)
        x = self.pos_enc(x)
        for j in range(self.N):
            x = self.layers[i](x, exp_out, source_mask, target_mask)
        return self.norm(x)

    

















def test():
    print("Just checking for tuntime errs!")

if __name__=="__main__":
    test()
    