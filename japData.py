import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd 
import spacy

# Lets import the data set
trainDf = pd.read_csv("dataset/train.csv")
valDf = pd.read_csv("dataset/val.csv")

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
    return [tokenize.text for tokenize in jap.tokenizer(sentences)]

def tokenize_en(sentences):
    return [tokenize.text for tokenize in en.tokenizer(sentences)]


def test():
    print("No RUNTIME Errs!")
if __name__ == "__main__":
    test()
    print(cleanData(trainDf))