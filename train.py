import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm 
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
import math
from japData import *
from model import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
N = 6
attnHeads = 8
ModelD = 512
BATCH_SIZE = 20



def trainModel(model, epochs, print_every=50):
    loop = tqdm(epochs) # for progress bars
     

    model.train()

    start = datetime.now()
    temp = start 

    Tloss = 0

    for epoch in range(epochs):
        for i, batch in enumerate(trainingIterator):
            src = batch.japanese.transpose(0,1)
            trg = batch.english.transpose(0,1)
            trg_input = trg[:,:-1]
            targets = trg[:,1:].contiguous().view(-1)
            
            # Use mask function to mask the above srcs and trgs
            src_mask, trg_mask = masks(src, trg_input)
            preds = model(src, trg_input, src_mask, trg_mask)
            optim.zero_grad()

            loss = F.cross_entropy(
                preds.view(-1, preds.size(-1)),
                targets,
                ignore_index=target_pad
            )
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
            if (i + 1) % print_every == 0:
                loss_avg = total_loss / print_every
                print("time = {}, epoch {}, iter = {}, loss = {}, {} per {} iters".format(
                    (datetime.now() - start) // 60,
                    epoch + 1,
                    i + 1,
                    loss_avg,
                    datetime.now() - temp,
                    print_every
                ))
                total_loss = 0
                temp = datetime.now()
                loop.set_postfix(loss=loss.item())
    
        print()

def main():
    source_vocab = len(japaneseText.vocab)
    target_vocab = len(englishText.vocab)
    model = Transfomer(source_vocab, target_vocab, ModelD, N, attnHeads).to(DEVICE)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    optim = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, factor=0.1, patience=10, verbose=True
    )
    #trainModel(model, EPOCHS)
    print(source_vocab)

if __name__ == "__name__":
   print(len(japaneseText.vocab))
   main()