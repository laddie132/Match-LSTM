#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
import torch.nn as nn
from models.layers import *
from utils.functions import answer_search


class MatchLSTMModel(torch.nn.Module):
    """
    match-lstm model for machine comprehension
    Args:
        - global_config: model_config with types dictionary

    Inputs:
        context: (batch, seq_len)
        question: (batch, seq_len)

    Outputs:
        answer_range: (batch, answer_len, context_len)
        vis_alpha: to show on visdom
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

        self_match_lstm_bidirection = global_config['model']['self_match_lstm_bidirection']
        self_match_lstm_direction_num = 2 if self_match_lstm_bidirection else 1
        self.enable_self_match = global_config['model']['self_match_lstm']

        self.enable_birnn_after_self = global_config['model']['birnn_after_self']

        encoder_word_layers = global_config['model']['encoder_word_layers']
        encoder_char_layers = global_config['model']['encoder_char_layers']

        self.init_ptr_hidden_mode = global_config['model']['init_ptr_hidden']
        hidden_mode = global_config['model']['hidden_mode']
        gated_attention = global_config['model']['gated_attention']
        self.enable_search = global_config['model']['answer_search']

        dropout_p = global_config['model']['dropout_p']

        # construct model
        self.embedding = GloveEmbedding(dataset_h5_path=global_config['data']['dataset_h5'])
        self.encoder = MyRNNBase(mode=hidden_mode,
                                 input_size=embedding_size,
                                 hidden_size=hidden_size,
                                 num_layers=encoder_word_layers,
                                 bidirectional=encoder_bidirection,
                                 dropout_p=dropout_p)
        encode_out_size = hidden_size * encoder_direction_num

        self.match_rnn = MatchRNN(mode=hidden_mode,
                                  input_size=encode_out_size,
                                  hidden_size=hidden_size,
                                  bidirectional=match_lstm_bidirection,
                                  gated_attention=gated_attention,
                                  dropout_p=dropout_p)
        match_lstm_out_size = hidden_size * match_lstm_direction_num

        if self.enable_self_match:
            self.self_match_rnn = MatchRNN(mode=hidden_mode,
                                           input_size=match_lstm_out_size,
                                           hidden_size=hidden_size,
                                           bidirectional=self_match_lstm_bidirection,
                                           gated_attention=gated_attention,
                                           dropout_p=dropout_p)
            match_lstm_out_size = hidden_size * self_match_lstm_direction_num

        if self.enable_birnn_after_self:
            self.birnn_after_self = MyRNNBase(mode=hidden_mode,
                                              input_size=match_lstm_out_size,
                                              hidden_size=hidden_size,
                                              num_layers=1,
                                              bidirectional=True,
                                              dropout_p=dropout_p)
            match_lstm_out_size = hidden_size * 2

        ptr_hidden_size = encode_out_size if self.init_ptr_hidden_mode == 'pooling' else hidden_size
        self.pointer_net = BoundaryPointer(mode=hidden_mode,
                                           input_size=match_lstm_out_size,
                                           hidden_size=ptr_hidden_size,  # just to fit init hidden on encoder generate
                                           dropout_p=dropout_p)

        # pointer net init hidden generate
        if self.init_ptr_hidden_mode == 'pooling':
            self.init_ptr_hidden = AttentionPooling(ptr_hidden_size)
        elif self.init_ptr_hidden_mode == 'linear':
            self.init_ptr_hidden = nn.Linear(match_lstm_out_size, ptr_hidden_size)
        elif self.init_ptr_hidden_mode == 'None':
            pass
        else:
            raise ValueError('Wrong init_ptr_hidden mode select %s, change to pooling or linear'
                             % self.init_ptr_hidden_mode)

    def forward(self, context, question):
        # get embedding: (seq_len, batch, embedding_size)
        context_vec, context_mask = self.embedding.forward(context)
        question_vec, question_mask = self.embedding.forward(question)

        # encode: (seq_len, batch, hidden_size)
        context_encode, context_new_mask = self.encoder.forward(context_vec, context_mask)
        question_encode, question_new_mask = self.encoder.forward(question_vec, question_mask)

        # match lstm: (seq_len, batch, hidden_size)
        qt_aware_ct, qt_aware_last_hidden, viz_alpha = self.match_rnn.forward(context_encode, context_new_mask,
                                                                              question_encode, question_new_mask)

        # self match lstm: (seq_len, batch, hidden_size)
        if self.enable_self_match:
            qt_aware_ct, qt_aware_last_hidden, _ = self.self_match_rnn.forward(qt_aware_ct, context_new_mask,
                                                                               qt_aware_ct, context_new_mask)

        # birnn after self match: (seq_len, batch, hidden_size)
        if self.enable_birnn_after_self:
            qt_aware_ct, _ = self.birnn_after_self.forward(qt_aware_ct, context_new_mask)

        # pointer net init hidden: (batch, hidden_size)
        ptr_net_hidden = None
        if self.init_ptr_hidden_mode == 'pooling':
            ptr_net_hidden = self.init_ptr_hidden.forward(question_encode, question_new_mask)
        elif self.init_ptr_hidden_mode == 'linear':
            ptr_net_hidden = self.init_ptr_hidden.forward(qt_aware_last_hidden)
            ptr_net_hidden = torch.tanh(ptr_net_hidden)

        # pointer net: (answer_len, batch, context_len)
        ans_range_prop = self.pointer_net.forward(qt_aware_ct, context_new_mask, ptr_net_hidden)
        ans_range_prop = ans_range_prop.transpose(0, 1)

        # answer range
        if self.enable_search:
            ans_range = answer_search(ans_range_prop, context_new_mask)
        else:
            ans_range = torch.max(ans_range_prop, 2)[1]

        return ans_range_prop, ans_range, viz_alpha
