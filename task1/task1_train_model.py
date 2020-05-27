import warnings
warnings.filterwarnings('ignore')

from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from dlnlputils.data import SparseFeaturesDataset
from dlnlputils.pipeline import train_eval_loop, predict_with_model, init_random_seed
import pickle
init_random_seed()

test_source = fetch_20newsgroups(subset='test')
train_source = fetch_20newsgroups(subset='train')
train_source_aug = train_source['data']
train_target_aug = train_source['target']

train_vectors = pickle.load(open('train_vectors.pkl', 'rb'))
test_vectors = pickle.load(open('test_vectors.pkl', 'rb'))

UNIQUE_LABELS_N = len(set(train_target_aug))
print('Количество уникальных меток', UNIQUE_LABELS_N)

train_dataset = SparseFeaturesDataset(train_vectors, train_target_aug)
test_dataset = SparseFeaturesDataset(test_vectors, test_source['target'])

with open('UNIQUE_WORDS_N.txt', 'r') as f:
    UNIQUE_WORDS_N = int(f.read())
    print("UNIQUE_WORDS_N=", UNIQUE_WORDS_N)


class model(nn.Module):
    def __init__(self, n_hidden_neurons):
        super(model, self).__init__()
        self.fc1 = torch.nn.Linear(UNIQUE_WORDS_N, n_hidden_neurons)
        self.act1 = torch.nn.Sigmoid()
        self.fc2 = torch.nn.Linear(n_hidden_neurons, UNIQUE_LABELS_N)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        return x

model = model(15)
scheduler = lambda optim: \
    torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=2, factor=0.5, verbose=True)

best_val_loss, best_model = train_eval_loop(model=model,
                                            train_dataset=train_dataset,
                                            val_dataset=test_dataset,
                                            criterion=F.cross_entropy,
                                            lr=1e-3,
                                            epoch_n=200,
                                            batch_size=32,
                                            l2_reg_alpha=0,
                                            lr_scheduler_ctor=scheduler,
                                            early_stopping_patience=3)

train_pred = predict_with_model(best_model, train_dataset)

train_loss = F.cross_entropy(torch.from_numpy(train_pred),
                             torch.from_numpy(np.asarray(train_target_aug)).long())


test_pred = predict_with_model(best_model, test_dataset)

test_loss = F.cross_entropy(torch.from_numpy(test_pred),
                            torch.from_numpy(test_source['target']).long())

from sklearn.metrics import precision_score, recall_score
def print_metrics(loss, y_true, pred):
    y_pred = pred.argmax(-1)
    print('Среднее значение функции потерь', float(loss))
    print('Доля верных ответов', accuracy_score(y_true, y_pred))
    print('Micro precision: ', precision_score(y_true, y_pred, average='micro'))
    print('Macro precision: ', precision_score(y_true, y_pred, average='macro'))
    print('Macro recall: ', recall_score(y_true, y_pred, average='micro'))
    print('Macro recall: ', recall_score(y_true, y_pred, average='macro'))

print('test')
print_metrics(test_loss, test_source['target'], test_pred)
print('-------------')
print('train')
print_metrics(train_loss, train_target_aug, train_pred)