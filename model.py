import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.legacy.data import Field, TabularDataset, BucketIterator, Iterator

import spacy
# We use spacy to load some desired languages

spacy_jp = spacy.load("ja_core_news_sm") # load japanese
spacy_en = spacy.load("en_core_web_sm") # load english
spacy_ge = spacy.load("de_core_news_sm") # load german

#def tokenize_jp(sometext):
 #   return [tok.sometext for tok in spacy_jp.tokenizer(sometext)]

def tokenize_en(sometext):
    return [tok.sometext for tok in spacy_en.tokenizer(sometext)]

def tokenize_ge(sometext):
    return [tok.sometext for tok in spacy_ge.tokenizer(sometext)]



def test():
    print("Just checking for tuntime errs!")

if __name__=="__main__":
    test()
