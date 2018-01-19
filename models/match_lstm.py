#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import sys
import yaml
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable

from models.layers import GloveEmbedding, MatchLSTM, BoundaryPointer


class MatchLSTMModel(torch.nn.Module):
    """
    match-lstm model for machine comprehension
    input: model_config with types dictionary
    """

    def __init__(self, global_config):
        super(MatchLSTMModel, self).__init__()
        embedding_size = global_config['model']['embedding_size']
        hidden_size = global_config['model']['hidden_size']
        encoder_bidirection = global_config['model']['encoder_bidirection']

        self.embedding = GloveEmbedding(glove_h5_path=global_config['data']['embedding_h5'])
        self.encoder = nn.LSTM(input_size=embedding_size,
                               hidden_size=hidden_size,
                               bidirectional=encoder_bidirection)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        pass

    def forward(self, context, question):
        pass