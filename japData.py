import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import spacy
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator
import torchtext
import gensim
from gensim.models import Doc2Vec
import tqdm
from gensim.models.doc2vec import TaggedDocument
from model import *
from datetime import datetime



# Lets import the data set
trainDf = pd.read_csv("dataset/train.csv")
valDf = pd.read_csv("dataset/val.csv")

# Some hyperparameters 

EPOCHS = 10
N = 6
attnHeads = 8
ModelD = 512
BATCH_SIZE = 20

# Our dataset is quite wild. Let's clean it up a bit

# Drop empty cells
trainDf.dropna(inplace=True)

# dataset is actually already quite clean so only had to drop na vals
def cleanData(dataset):
    cleanset = dataset.dropna
    return cleanset


spacy_jp = spacy.load("ja_core_news_sm") # load japanese
spacy_en = spacy.load("en_core_web_sm") # load english
# Now we toenize the sets
def tokenize_jp(sentences):
    return [tokenize.text for tokenize in spacy_jp.tokenizer(sentences)]

def tokenize_en(sentences):
    return [tokenize.text for tokenize in spacy_en.tokenizer(sentences)]


# We have to implement the data fields so we can build a vocabulary based off token
japaneseText = Field(tokenize=tokenize_jp)
englishText = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>')

dataField = [('japanese', japaneseText), ('english', englishText)]

cleanTrainDf = cleanData(trainDf)
cleanValDf = cleanData(valDf)

train, val = TabularDataset.splits(
    path='dataset/',
    train='train.csv',
    validation='val.csv',
    format='csv',
    fields=dataField
)

japaneseText.build_vocab(train, val)
englishText.build_vocab(train, val)

trainingIterator = BucketIterator(
    train,
    batch_size=BATCH_SIZE,
    sort_key=lambda a: len(a.english),
    shuffle=True
)

def masks(input_sequence, target_sequence):
    input_padding = japaneseText.vocab.stoi['<pad>']
    input_masking = (input_sequence != input_padding).unsqueeze(1)

    target_padding = englishText.vocab.stoi['<pad>']
    target_masking = (target_sequence != target_padding).unsqueeze(1)
    size = target_sequence.size(1)
    occludedMask = np.triu(np.ones((1, size, size)), k=1).astype(np.uint8)
    occludedMask = torch.autograd.Variable(torch.from_numpy(occludedMask) == 0).to(device=DEVICE)
    target_masking = target_masking & occludedMask

    return input_masking, target_masking

def trainModel(model, epochs, print_every=50):
    #loop = tqdm(epochs) # for progress bars
     

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
                #loop.set_postfix(loss=loss.item())
    
        print()

def main():
    source_vocab = len(japaneseText.vocab)
    target_vocab = len(englishText.vocab)
    model = Transfomer(source_vocab, target_vocab, ModelD, N, attnHeads).to(DEVICE)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.1, patience=10, verbose=True
    )
    trainModel(model, EPOCHS)
    #print(source_vocab)

def test():
    batch = next(iter(trainingIterator))
    print(batch.english)
    print("No RUNTIME Errs!")
if __name__ == "__main__":
    #print(tokenize_jp(trainDf))
    #print(len(japaneseText.vocab))
    main()
    