import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import spacy
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator


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
cleanvalDf = cleanData(valDf)
japaneseText = build_vocab(cleanTrainDf, cleanValDf)
englishText = build_vocab(cleanTrainDf, cleanValDf)

trainingIterator = BucketIterator(
    cleanTrainDf,
    batch_size=BATCH_SIZE,
    sort_key=lambda a: len(a.english),
    shuffle=True
)


def test():
    print("No RUNTIME Errs!")
if __name__ == "__main__":
    #print(tokenize_jp(trainDf))
    batch = next(iter(trainingIterator))
    print(batch.english)
    test()
    