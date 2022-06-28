import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
import torch.nn as nn
import torch.optim as optim

# We're going to implemtn a general transformer architecture


class Transfomer(nn.Module):
    def __init__(self, embedding_size, source_vocab_size, target_vocab_size, source_pad_idx, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout, maximum_seq_len, device):
        super(Transfomer, self).__init__()
        self.source_word_embedding = nn.Embedding(source_vocab_size, embedding_size)
        self.source_position_embedding = nn.Embedding(maximum_seq_len, embedding_size) #pe
        self.target_word_embedding = nn.Embedding(target_vocab_size, embedding_size)
        self.device=device
        self.transformer = nn.Transfomer(embedding_size, num_heads, num_encoder_layers, num_decoder_layers, forward_expansion, dropout)
        self.fullyconnected_out = nn.Linear(embedding_size, target_vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.source_pad_idx = source_pad_idx
        # refer to transformer paper


    def source_mask(self, source):
        source_mask = source.transpose(0,1) == self.source_pad_idx
        #(N, SOURCE_LEN)
        return source_mask.to(self.device)

    def forward(self, source, target):
        source_sequence_len, N = source.shape
        target_sequence_len, N = target.shape

        source_positions = (
            torch.arange(0, source_sequence_len)
            .unsqueeze(1)
            .expand(source_sequence_len, N)
            .to(self.device)
        ) 
        target_positions = (
            torch.arange(0, target_sequence_len)
            .unsqueeze(1)
            .expand(target_sequence_len, N)
            .to(self.device)
        ) 
        embed_source = self.dropout(
            (self.source_word_embedding(source) + self.source_position_embedding(source_positions))
        )
        embed_target = self.dropout(
            (self.target_word_embedding(target) + self.target_position_embedding(target_positions))
        )
        source_padding_mask = self.source_mask(source)
        target_mask = self.transformer.generate_square_subsequent_mask(target_sequence_len).to(
            self.device
        )
        out = self.transformer(embed_source, embed_target, source_key_padding_mask=source_padding_mask, tag_mask=target_mask)
        out = self.fullyconnected_out(out)
        return out




def test():
    print("Just checking for tuntime errs!")

if __name__=="__main__":
    test()
       