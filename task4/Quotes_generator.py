import numpy as np
import torch, torch.nn as nn
from tqdm import tqdm
from model import CharRNN
from get_batches import get_batches


file = 'data/dante.txt'
def read_file(file):
    import chardet

    rawdata = open(file, "br").read()
    result = chardet.detect(rawdata)
    charenc = result['encoding']
    return open(file, 'r', encoding=charenc).read()

quotes = read_file(file)
tokens = list(set(''.join(quotes)))
print(tokens)
token_to_id = {token: idx for idx, token in enumerate(tokens)}
id_to_token = {idx: token for token, idx in token_to_id.items()}
num_tokens = len(tokens)
# encode the text
encoded = np.array([token_to_id[ch] for ch in quotes])

# check if GPU is available
train_on_gpu = torch.cuda.is_available()
if(train_on_gpu):
    print('Training on GPU!')
else:
    print('No GPU available, training on CPU; consider making n_epochs very small.')


# define and print the net
n_hidden=512
n_layers=4

net = CharRNN(num_tokens=num_tokens, n_hidden=n_hidden, n_layers=n_layers)
print(net)


def train(net, data, token_to_id=token_to_id, epochs=10, batch_size=10, seq_length=50, lr=0.001, clip=5, val_frac=0.1, print_every=10):
    ''' Training a network

        Arguments
        ---------

        net: CharRNN network
        data: text data to train the network
        epochs: Number of epochs to train
        batch_size: Number of mini-sequences per mini-batch, aka batch size
        seq_length: Number of character steps per mini-batch
        lr: learning rate
        clip: gradient clipping
        val_frac: Fraction of data to hold out for validation
        print_every: Number of steps for printing training and validation loss

    '''
    net.train()

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # create training and validation data
    val_idx = int(len(data) * (1 - val_frac))
    data, val_data = data[:val_idx], data[val_idx:]

    if (train_on_gpu):
        net.cuda()

    counter = 0
    for e in tqdm(range(epochs), total=epochs):
        # initialize hidden state
        h = net.init_hidden(batch_size)

        for x, y in get_batches(data, token_to_id, batch_size, seq_length):
            counter += 1

            inputs, targets = torch.from_numpy(x), torch.from_numpy(y)

            if (train_on_gpu):
                inputs, targets = inputs.cuda(), targets.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            net.zero_grad()

            # get the output from the model
            output, h = net(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output, targets.view(batch_size * seq_length))
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(net.parameters(), clip)
            opt.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = net.init_hidden(batch_size)
                val_losses = []
                net.eval()
                for x, y in get_batches(val_data, token_to_id, batch_size, seq_length):
                    x, y = torch.from_numpy(x), torch.from_numpy(y)

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    inputs, targets = x, y
                    if (train_on_gpu):
                        inputs, targets = inputs.cuda(), targets.cuda()

                    output, val_h = net(inputs, val_h)
                    val_loss = criterion(output, targets.view(batch_size * seq_length))

                    val_losses.append(val_loss.item())

                net.train()  # reset to train mode after iterationg through validation data

                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.4f}...".format(loss.item()),
                      "Val Loss: {:.4f}".format(np.mean(val_losses)))


batch_size = 64
seq_length = 160 #max length verses
n_epochs = 50 # start smaller if you are just testing initial behavior

# train the model
train(net, encoded, epochs=n_epochs, batch_size=batch_size, seq_length=seq_length, lr=0.001, print_every=10)

model_dante = 'rnn_50_epoch.net'

checkpoint = {'n_hidden': net.n_hidden,
              'n_layers': net.n_layers,
              'state_dict': net.state_dict()}

with open(model_dante, 'wb') as f:
    torch.save(checkpoint, f)