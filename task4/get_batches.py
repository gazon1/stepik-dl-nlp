import numpy as np


def get_batches(arr, token_to_id, batch_size, seq_length):
    '''В коллекции один документ. Напр., коллекция - это роман Толстого "Война и мир"
       Create a generator that returns batches of size
       batch_size x seq_length from arr.

       Arguments
       ---------
       arr: Array you want to make batches from
       batch_size: Batch size, the number of sequences per batch
       seq_length: Number of encoded chars in a sequence
    '''

    batch_size_total = batch_size * seq_length
    # total number of batches we can make, // integer division, round down
    n_batches = len(arr) // batch_size_total

    # Keep only enough characters to make full batches
    arr = arr[:n_batches * batch_size_total]
    # Reshape into batch_size rows, n. of first row is the batch size, the other lenght is inferred
    arr = arr.reshape((batch_size, -1))

    # iterate through the array, one sequence at a time
    for n in range(0, arr.shape[1], seq_length):
        # The features
        x = arr[:, n:n + seq_length]
        # The targets, shifted by one
        y = np.zeros_like(x) + token_to_id[' ']
        try:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, n + seq_length]
        except IndexError:
            y[:, :-1], y[:, -1] = x[:, 1:], arr[:, 0]
        yield x, y

    # when we call get batches we are going
    # to create a generator that iteratest through our array and returns x, y with yield command

def get_batches_many_docs_in_collection(data, token_to_id, max_len=None, dtype='int32', batch_first=True):
    '''В коллекции много документов - напр., коллекция - это список русских имен
        и задача выучить распределение следующей буквы в зависимости от предыдущей в этой коллекции

       Casts a list of docs into rnn-digestable matrix
    '''

    # TODO: переписать в виде итератора yield
    max_len = max_len or max(map(len, data))
    data_ix = np.zeros([len(data), max_len], dtype) + token_to_id[' ']

    for i in range(len(data)):
        line_ix = [token_to_id[c] for c in data[i]]
        data_ix[i, :len(line_ix)] = line_ix

    if not batch_first:  # convert [batch, time] into [time, batch]
        data_ix = np.transpose(data_ix)

    return data_ix