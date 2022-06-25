import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
import math
from japData import *

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 10
N = 6
attnHeads = 8
ModelD = 512
BATCH_SIZE = 20


def trainModel(model, epochs, print_every=50):
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
    
        print()


model.to(device=DEVICE)
#train(model, EPOCHS)