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

        # set config
        embedding_size = global_config['model']['embedding_size']
        self.hidden_size = global_config['model']['hidden_size']
        self.enable_cuda = global_config['train']['enable_cuda']

        encoder_bidirection = global_config['model']['encoder_bidirection']
        self.encoder_direction_num = 1
        if encoder_bidirection:
            self.encoder_direction_num = 2

        match_lstm_bidirection = global_config['model']['match_lstm_bidirection']
        self.match_lstm_direction_num = 1
        if match_lstm_bidirection:
            self.match_lstm_direction_num = 2

        # construct model
        self.embedding = GloveEmbedding(dataset_h5_path=global_config['data']['dataset_h5'])
        self.encoder = nn.LSTM(input_size=embedding_size,
                               hidden_size=self.hidden_size,
                               bidirectional=encoder_bidirection)
        encode_out_size = self.hidden_size * self.encoder_direction_num

        self.match_lstm = MatchLSTM(input_size=encode_out_size,
                                    hidden_size=self.hidden_size,
                                    bidirectional=match_lstm_bidirection,
                                    enable_cuda=self.enable_cuda)
        match_lstm_out_size = self.hidden_size * self.match_lstm_direction_num

        self.pointer_net = BoundaryPointer(input_size=match_lstm_out_size,
                                           hidden_size=self.hidden_size,
                                           enable_cuda=self.enable_cuda)

        # pointer net init hidden generate
        self.ptr_net_hidden_linear = nn.Linear(match_lstm_out_size, self.hidden_size)

    def forward(self, context, question):
        batch_size = context.shape[0]

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

        # packed padding values
        context_vec_pack = torch.nn.utils.rnn.pack_padded_sequence(context_vec, c_vin_length)
        question_vec_pack = torch.nn.utils.rnn.pack_padded_sequence(question_vec, q_vin_length)

        # encode
        encode_hidden = init_hidden(self.encoder_direction_num, batch_size, self.hidden_size, self.enable_cuda)
        context_encode_pack, _ = self.encoder.forward(context_vec_pack, encode_hidden)
        question_encode_pack, _ = self.encoder.forward(question_vec_pack, encode_hidden)

        # pad values
        context_encode, context_length = torch.nn.utils.rnn.pad_packed_sequence(context_encode_pack)
        question_encode, question_length = torch.nn.utils.rnn.pad_packed_sequence(question_encode_pack)

        # match lstm
        qt_aware_ct = self.match_lstm.forward(context_encode, context_length, question_encode, question_length)

        # pointer net
        qt_aware_last_hidden = qt_aware_ct[-1, :]
        ptr_net_hidden = self.ptr_net_hidden_linear.forward(qt_aware_last_hidden)
        answer_range = self.pointer_net.forward(qt_aware_ct, ptr_net_hidden)

        return answer_range.transpose(0, 1)