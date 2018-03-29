#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def init_hidden(num_layers_directions, batch, hidden_size, enable_cuda):
    """
    - notice: replaced by function `tensor.new.zero_()`
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
    - notice: replaced by function `tensor.new.zero_()`
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


def to_long_variable(np_array, enable_cuda=False):
    """
    convert a numpy array to Torch Variable with LongTensor
    :param np_array:
    :param enable_cuda:
    :return:
    """
    if enable_cuda:
        return Variable(torch.from_numpy(np_array).type(torch.LongTensor)).cuda()

    return Variable(torch.from_numpy(np_array).type(torch.LongTensor))


def to_long_tensor(np_array):
    """
    convert to long torch tensor
    :param np_array:
    :return:
    """
    return torch.from_numpy(np_array).type(torch.LongTensor)


def to_variable(tensor, enable_cuda=False):
    """
    convert to torch variable
    :param tensor:
    :param enable_cuda:
    :return:
    """
    if enable_cuda:
        return Variable(tensor).cuda()

    return Variable(tensor)


class MyNLLLoss(torch.nn.modules.loss._Loss):
    """
    a standard negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    Shape:
        - y_pred: (batch, answer_len, prob)
        - y_true: (batch, answer_len)       # todo: end postion should also include
        - output: loss
    """
    def __init__(self):
        super(MyNLLLoss, self).__init__()

    def forward(self, y_pred, y_true):
        torch.nn.modules.loss._assert_no_grad(y_true)

        a = Variable(torch.LongTensor([[0, 1]]))
        if y_true.is_cuda:
            a = a.cuda()
        y_true = y_true - a                     # temp

        y_pred_log = torch.log(y_pred)
        loss = []
        for i in range(y_pred.shape[0]):
            tmp_loss = F.nll_loss(y_pred_log[i], y_true[i], reduce=False)
            one_loss = tmp_loss[0] + tmp_loss[1]
            loss.append(one_loss)

        loss_stack = torch.stack(loss)
        return torch.mean(loss_stack)


def count_parameters(model):
    """
    get parameters count that require grad
    :param model:
    :return:
    """
    parameters_num = 0
    for par in model.parameters():
        if not par.requires_grad:
            continue

        tmp_par_shape = par.size()
        tmp_par_size = 1
        for ele in tmp_par_shape:
            tmp_par_size *= ele
        parameters_num += tmp_par_size
    return parameters_num


def compute_mask(v, padding_idx=0):
    mask = torch.ne(v, padding_idx).float()
    return mask


def generate_mask(batch_length, enable_cuda=False):
    sum_one = np.sum(np.array(batch_length))

    one = Variable(torch.ones(int(sum_one)))
    if enable_cuda:
        one = one.cuda()

    mask_packed = torch.nn.utils.rnn.PackedSequence(one, batch_length)
    mask, _ = torch.nn.utils.rnn.pad_packed_sequence(mask_packed)

    return mask


def masked_softmax(x, m=None, dim=-1):
    '''
    Softmax with mask (optional)
    '''
    if m is not None:
        m = m.float()
        x = x * m
    e_x = torch.exp(x - torch.max(x, dim=dim, keepdim=True)[0])
    if m is not None:
        e_x = e_x * m
    softmax = e_x / (torch.sum(e_x, dim=dim, keepdim=True) + 1e-6)
    return softmax