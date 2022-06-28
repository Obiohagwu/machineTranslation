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
from datetime import datetimer