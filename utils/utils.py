#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def init_hidden(num_layers_directions, batch, hidden_size):
    """
    lstm init hidden out, state
    :param num_layers_directions: num_layers \* num_directions
    :param batch: 
    :param hidden_size: 
    :return: 
    """

    return (Variable(torch.zeros(num_layers_directions, batch, hidden_size)),
            Variable(torch.zeros(num_layers_directions, batch, hidden_size)))


def init_hidden_cell(batch, hidden_size):
    """
    lstm init hidden out, state
    :param batch:
    :param hidden_size:
    :return:
    """

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