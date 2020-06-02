from torch import nn
import torch.nn.functional as F
import torch


class CharRNN(nn.Module):
    def __init__(self, num_tokens, emb_size=16, n_hidden=300, drop_prob=0.5, n_layers=2):
        super(self.__class__, self).__init__()
        self.emb = nn.Embedding(num_tokens, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size=n_hidden, num_layers=n_layers,
                            dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.hid_to_logits = nn.Linear(n_hidden, num_tokens)

        self.n_layers = n_layers
        self.n_hidden = n_hidden

    def forward(self, x, hidden=None):
        # assert isinstance(x, Variable) and isinstance(x.data, torch.LongTensor)
        ## TODO: Get the outputs and the new hidden state from the lstm
        x = self.emb(x)
        #         print(x.shape)
        if hidden is None:
            r_output, hidden = self.lstm(x)
        else:
            r_output, hidden = self.lstm(x, hidden)

        ## TODO: pass through a dropout layer
        out = self.dropout(r_output)

        # Stack up LSTM outputs using view
        # you may need to use contiguous to reshape the output
        out = out.contiguous().view(-1, self.n_hidden)

        ## TODO: put x through the fully-connected layer
        next_logits = self.hid_to_logits(out)
        next_logp = F.log_softmax(next_logits, dim=-1)

        # return the final output and the hidden state
        return next_logp, hidden

    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x n_hidden,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        if (torch.cuda.is_available()):
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_().cuda())
        else:
            hidden = (weight.new(self.n_layers, batch_size, self.n_hidden).zero_(),
                      weight.new(self.n_layers, batch_size, self.n_hidden).zero_())

        return hidden