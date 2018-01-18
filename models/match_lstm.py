#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import sys
import yaml
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable

from models.layers import GloveEmbedding


class MatchLSTM(torch.nn.Module):
    """
    match-lstm model for machine comprehension
    input: model_config with types dictionary
    """

    def __init__(self, global_config):
        super(MatchLSTM, self).__init__()

        self.embedding = GloveEmbedding(glove_h5_path=global_config['data']['embedding_h5'])
        self.encoder = nn.LSTM()

    def forward(self, context, question):
        pass