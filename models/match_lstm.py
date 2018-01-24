#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import sys
import yaml
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
from utils.utils import init_hidden
from models.layers import GloveEmbedding, MatchLSTM, BoundaryPointer


class MatchLSTMModel(torch.nn.Module):
    """
    match-lstm model for machine comprehension
    Args:
        - global_config: model_config with types dictionary

    Inputs:
        context: (batch, seq_len)
        question: (batch, seq_len)

    Outputs:
        answer_range: (batch, answer_len, prob)
    """

    def __init__(self, global_config):
        super(MatchLSTMModel, self).__init__()
        embedding_size = global_config['model']['embedding_size']
        hidden_size = global_config['model']['hidden_size']
        encoder_bidirection = global_config['model']['encoder_bidirection']

        self.hidden_size = hidden_size
        self.embedding = GloveEmbedding(glove_h5_path=global_config['data']['embedding_h5'])
        self.encoder = nn.LSTM(input_size=embedding_size,
                               hidden_size=hidden_size,
                               bidirectional=encoder_bidirection)
        encode_out_size = hidden_size
        if encoder_bidirection:
            encode_out_size *= 2
        self.match_lstm = MatchLSTM(input_size=encode_out_size,
                                    hidden_size=hidden_size,
                                    bidirectional=True)
        self.pointer_net = BoundaryPointer(input_size=hidden_size*2,
                                           hidden_size=hidden_size)

    def forward(self, context, question):
        batch_size = context.shape[0]
        hidden = init_hidden(1, batch_size, self.hidden_size)

        context_vec = self.embedding.forward(context).transpose(0, 1)
        question_vec = self.embedding.forward(question).transpose(0, 1)

        context_encode, _ = self.encoder.forward(context_vec, hidden)
        question_encode, _ = self.encoder.forward(question_vec, hidden)

        qt_aware_ct = self.match_lstm.forward(context_encode, question_encode)
        answer_range = self.pointer_net.forward(qt_aware_ct)

        return answer_range.transpose(0, 1)