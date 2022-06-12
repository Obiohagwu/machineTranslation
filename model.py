import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

# We use spacy to load some desired languages

spacy_jp = spacy.load("jp") # load japanese
spacy_en = spacy.load("en") # load english
spacy_ge = spacy.load("ge") # load german

def tokenize_jp(sometext):
    return [tok.sometext for tok in spacy_jp.tokenizer(sometext)]

def tokenize_en(sometext):
    return [tok.sometext for tok in spacy_en.tokenizer(sometext)]

def tokenize_ge(sometext):
    return [tok.sometext for tok in spacy_ge.tokenizer(sometext)]



def test():
    print("Just checking for tuntime errs!")

if __name__=="__main__":
    test()
