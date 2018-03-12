#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def init_hidden(num_layers_directions, batch, hidden_size, enable_cuda):
    """
    lstm init hidden out, state
    :param num_layers_directions: num_layers \* num_directions
    :param batch: 
    :param hidden_size: 
    :return: 
    """
    if enable_cuda:
        return (Variable(torch.zeros(num_layers_directions, batch, hidden_size)).cuda(),
                Variable(torch.zeros(num_layers_directions, batch, hidden_size)).cuda())

    return (Variable(torch.zeros(num_layers_directions, batch, hidden_size)),
            Variable(torch.zeros(num_layers_directions, batch, hidden_size)))


def init_hidden_cell(batch, hidden_size, enable_cuda):
    """
    lstm init hidden out, state
    :param batch:
    :param hidden_size:
    :return:
    """
    if enable_cuda:
        return (Variable(torch.zeros(batch, hidden_size)).cuda(),
                Variable(torch.zeros(batch, hidden_size)).cuda())

    return (Variable(torch.zeros(batch, hidden_size)),
            Variable(torch.zeros(batch, hidden_size)))


def pad_sequences(sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
    """
    FROM KERAS
    Pads each sequence to the same length:
    the length of the longest sequence.
    If maxlen is provided, any sequence longer
    than maxlen is truncated to maxlen.
    Truncation happens off either the beginning (default) or
    the end of the sequence.
    Supports post-padding and pre-padding (default).
    # Arguments
        sequences: list of lists where each element is a sequence
        maxlen: int, maximum length
        dtype: type to cast the resulting sequence.
        padding: 'pre' or 'post', pad either before or after each sequence.
        truncating: 'pre' or 'post', remove values from sequences larger than
            maxlen either in the beginning or in the end of the sequence
        value: float, value to pad the sequences to the desired value.
    # Returns
        x: numpy array with dimensions (number_of_sequences, maxlen)
    """
    lengths = [len(s) for s in sequences]

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x


def packed_array(vin, padding_idx=0, enable_cuda=False):
    """
    transform a numpy array with padding to packed squences
    :param vin: (batch, max_seq_len)
    :param padding_idx: the existing padding idx in variable 'vin'
    :param enable_cuda:
    :return:
    """
    def real_len_no_pad(narray):
        n = 0
        i = len(narray) - 1
        while i >= 0:
            if narray[i] == padding_idx:
                n += 1
            else:
                break

            i -= 1
        return len(narray) - n

    vin_len = map(lambda v: real_len_no_pad(v), vin)
    vin_len = np.array(list(vin_len))
    vin_with_len = np.column_stack((vin, vin_len))
    vin_with_len_sorted = sorted(vin_with_len, key=lambda v: v[len(v)-1], reverse=True)
    vin_with_len_sorted = np.array(vin_with_len_sorted)

    ldx = vin.shape[1]
    vin_sorted = convert_long_variable(vin_with_len_sorted[:, :ldx], enable_cuda)
    len_sorted = vin_with_len_sorted[:, ldx:].T[0]

    pack = torch.nn.utils.rnn.pack_padded_sequence(vin_sorted, len_sorted, batch_first=True)

    return pack


def convert_long_variable(np_array, enable_cuda=False):
    if enable_cuda:
        return torch.autograd.Variable(torch.from_numpy(np_array).type(torch.LongTensor)).cuda()

    return torch.autograd.Variable(torch.from_numpy(np_array).type(torch.LongTensor))


class MyNLLLoss(torch.nn.modules.loss._Loss):
    """
    a standard negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    Shape:
        - y_pred: (batch, answer_len, prob)
        - y_true: (batch, answer_len)
        - output: loss
    """
    def __init__(self):
        super(MyNLLLoss, self).__init__()

    def forward(self, y_pred, y_true):
        torch.nn.modules.loss._assert_no_grad(y_true)

        y_pred_log = torch.log(y_pred)
        loss = []
        for i in range(y_pred.shape[0]):
            tmp_loss = F.nll_loss(y_pred_log[i], y_true[i], reduce=False)
            one_loss = F.mul(tmp_loss[0], tmp_loss[1])
            loss.append(one_loss)

        loss_stack = torch.stack(loss)
        return torch.mean(loss_stack)