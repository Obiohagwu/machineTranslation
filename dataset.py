from logging import root
from posixpath import split

from attr import fields
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator

import spacy
import spacy
# We use spacy to load some desired languages

spacy_jp = spacy.load("ja_core_news_sm") # load japanese
spacy_en = spacy.load("en_core_web_sm") # load english
spacy_ge = spacy.load("de_core_news_sm") # load german

def tokenize_jp(sometext):
    return [tok.sometext for tok in spacy_jp.tokenizer(sometext)] # returns the tokenised version of strings in spacy_jp

def tokenize_en(sometext):
    return [tok.sometext for tok in spacy_en.tokenizer(sometext)] # returns the tokenised version of strings in spacy_en

def tokenize_ge(sometext):
    return [tok.sometext for tok in spacy_ge.tokenizer(sometext)] # returns the tokenised version of strings in spacy_ge



german = Field(tokenize=tokenize_jp, lower=True, init_token="<sos>", eos_token="<eos>")

english = Field(tokenize=tokenize_en, lower=True, init_token="<sos>", eos_token="<eos>")

japanese = Field(tokenize=tokenize_jp, lower=True, init_token="<sos>", eos_token="<eos>")

# Now we split files into train and test sets

train, valid, test = Multi30k(root=".data", split=('train', 'valid', 'test'), language_pair=('en', 'de')))
german.build_vocab(train, max_size=10000, min_freq=2)
english.build_vocab(train, max_size=10000, min_freq=2)


def test():
    print("Just checking for tuntime errs!")

if __name__=="__main__":
    test()