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
from torch.utils.tensorboard import SummaryWriter
from gensim.models.doc2vec import TaggedDocument


from modelTransformer import Transfomer


# Lets import the data set
trainDf = pd.read_csv("dataset/train.csv")
valDf = pd.read_csv("dataset/val.csv")



EPOCHS = 10
N = 6
attnHeads = 8
ModelD = 512
BATCH_SIZE = 20

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


train_data, valid_data = TabularDataset.splits(
    path='dataset/',
    train='train.csv',
    validation='val.csv',
    format='csv',
    fields=dataField
)

japaneseText.build_vocab(train_data, valid_data)
englishText.build_vocab(train_data, valid_data)

trainingIterator = BucketIterator(
    train_data,
    batch_size=BATCH_SIZE,
    sort_key=lambda a: (len(a.japanese),len(a.english)),
    shuffle=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

load_model = True
save_model = True

# some training hyperparams
num_epochs = 10000
learning_rate = 3e-4
batch_size = 32

# some model hyperparameters

source_vocab_size = len(japanese.vocab)
target_vocab_size = len(english.vocab)
embedding_size = 512
num_heads = 8 # attn heads
num_decoder_layers=3
num_encoder_layers=3
dropout=0.1
maximum_seq_len=100
forward_expansion=4
source_pad_idx=english.vocab.stoi["<pad>"]

# to visual loss vs runs

visual = SummaryWriter("runs/loss")

step=9
# train fucn

def trainModel(model, epochs):
    pass



if __name__=="__main__":
    #main()
    batch = next(iter(trainingIterator))
    print(batch.japanese)