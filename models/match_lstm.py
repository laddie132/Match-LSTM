#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import sys
import yaml
import torch
import torch.nn as nn
import torch.functional as F
from torch.autograd import Variable
from utils.utils import *
from dataset.preprocess_data import PreprocessData
from models.layers import *


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

        # set config
        embedding_size = global_config['model']['embedding_size']
        hidden_size = global_config['model']['hidden_size']
        self.enable_cuda = global_config['train']['enable_cuda']

        encoder_bidirection = global_config['model']['encoder_bidirection']
        encoder_direction_num = 2 if encoder_bidirection else 1

        match_lstm_bidirection = global_config['model']['match_lstm_bidirection']
        match_lstm_direction_num = 2 if match_lstm_bidirection else 1

        dropout_p = global_config['model']['dropout_p']
        self.dropout = torch.nn.Dropout(p=dropout_p)

        # construct model
        self.embedding = GloveEmbedding(dataset_h5_path=global_config['data']['dataset_h5'],
                                        dropout_p=0.)
        self.encoder = MyLSTM(input_size=embedding_size,
                              hidden_size=hidden_size,
                              bidirectional=encoder_bidirection,
                              dropout_p=dropout_p)
        encode_out_size = hidden_size * encoder_direction_num

        self.match_lstm = MatchLSTM(input_size=encode_out_size,
                                    hidden_size=hidden_size,
                                    bidirectional=match_lstm_bidirection,
                                    enable_cuda=self.enable_cuda)
        match_lstm_out_size = hidden_size * match_lstm_direction_num

        self.pointer_net = BoundaryPointer(input_size=match_lstm_out_size,
                                           hidden_size=hidden_size)

        # pointer net init hidden generate
        self.ptr_net_hidden_linear = nn.Linear(match_lstm_out_size, hidden_size)

    def forward(self, context, question):
        # get sorted length
        c_vin, c_vin_length = sort_length(context.data.cpu().numpy(),
                                          padding_idx=PreprocessData.padding_idx,
                                          enable_cuda=self.enable_cuda)
        q_vin, q_vin_length = sort_length(question.data.cpu().numpy(),
                                          padding_idx=PreprocessData.padding_idx,
                                          enable_cuda=self.enable_cuda)

        # get embedding
        context_vec = self.embedding.forward(context).transpose(0, 1)
        question_vec = self.embedding.forward(question).transpose(0, 1)

        # encode
        context_encode, context_length = self.encoder.forward(context_vec, c_vin_length)
        question_encode, question_length = self.encoder.forward(question_vec, q_vin_length)

        # match lstm
        qt_aware_ct = self.match_lstm.forward(context_encode, context_length, question_encode, question_length)

        # pointer net
        qt_aware_last_hidden = qt_aware_ct[-1, :]
        ptr_net_hidden = self.ptr_net_hidden_linear.forward(qt_aware_last_hidden)
        ptr_net_hidden = torch.tanh(ptr_net_hidden)
        answer_range = self.pointer_net.forward(qt_aware_ct, ptr_net_hidden)

        return answer_range.transpose(0, 1)