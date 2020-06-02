import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils import clip_grad_norm_
from data_utils import Dictionary, Corpus
from tqdm import tqdm

# Device configuration
if not torch.cuda.is_available():
    raise RuntimeError
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device:", device)

# Hyper-parameters
embed_size = 128
hidden_size = 1024
num_layers = 1
num_epochs = 5
num_samples = 1000  # number of words to be sampled
batch_size = 20
seq_length = 30
learning_rate = 0.002

# Load "Penn Treebank" dataset
corpus = Corpus()
ids = corpus.get_data('data/train.txt', batch_size)
vocab_size = len(corpus.dictionary)
num_batches = ids.size(1) // seq_length