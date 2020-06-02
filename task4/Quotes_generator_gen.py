import torch, torch.nn as nn
import torch.nn.functional as F
from model import CharRNN
import numpy as np



# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else:
    print('No GPU available, training on CPU; consider making n_epochs very small.')

file = 'data/dante.txt'
def read_file(file):
    import chardet

    rawdata = open(file, "br").read()
    result = chardet.detect(rawdata)
    charenc = result['encoding']
    return open(file, 'r', encoding=charenc).read()

quotes = read_file(file)
tokens = list(set(''.join(quotes)))
token_to_id = {token: idx for idx, token in enumerate(tokens)}
id_to_token = {idx: token for token, idx in token_to_id.items()}
num_tokens = len(tokens)
# encode the text
encoded = np.array([token_to_id[ch] for ch in quotes])


# Here we have loaded in a model that trained over 2 epochs `rnn_20_epoch.net`
with open('rnn_50_epoch.net', 'rb') as f:
    checkpoint = torch.load(f)

loaded = CharRNN(num_tokens, n_hidden=checkpoint['n_hidden'], n_layers=checkpoint['n_layers'])
loaded.load_state_dict(checkpoint['state_dict'])

def predict(net, char, h=None, top_k=None):
    ''' Given a character, predict the next character.
        Returns the predicted character and the hidden state.
    '''

    # tensor inputs
    x = np.array([[token_to_id[char]]])
    inputs = torch.from_numpy(x)

    if (train_on_gpu):
        inputs = inputs.cuda()

    # detach hidden state from history
    h = tuple([each.data for each in h])
    # get the output of the model
    out, h = net(inputs, h)

    # get the character probabilities
    # apply softmax to get p probabilities for the likely next character giving x
    p = F.softmax(out, dim=1).data
    if (train_on_gpu):
        p = p.cpu()  # move to cpu

    # get top characters
    # considering the k most probable characters with topk method
    if top_k is None:
        top_ch = np.arange(num_tokens)
    else:
        p, top_ch = p.topk(top_k)
        top_ch = top_ch.numpy().squeeze()

    # select the likely next character with some element of randomness
    p = p.numpy().squeeze()
    char = np.random.choice(top_ch, p=p / p.sum())

    # return the encoded value of the predicted char and the hidden state
    return id_to_token[char], h

def sample(net, size, prime='Il', top_k=None):
    if (train_on_gpu):
        net.cuda()
    else:
        net.cpu()

    net.eval()  # eval mode

    # First off, run through the prime characters
    chars = [ch for ch in prime]
    h = net.init_hidden(1)
    for ch in prime:
        char, h = predict(net, ch, h, top_k=top_k)

    chars.append(char)

    # Now pass in the previous character and get a new one
    for ii in range(size):
        char, h = predict(net, chars[-1], h, top_k=top_k)
        chars.append(char)

    return ''.join(chars)

print(sample(loaded, 800, top_k=5, prime='Nel'))

