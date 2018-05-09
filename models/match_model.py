#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
import torch.nn as nn
from models.layers import *
from dataset.preprocess_data import PreprocessData
from utils.functions import answer_search


class MatchLSTMModel(torch.nn.Module):
    """
    match-lstm model for machine comprehension
    Args:
        - global_config: model_config with types dictionary

    Inputs:
        context: (batch, seq_len)
        question: (batch, seq_len)
        context_char: (batch, seq_len, word_len)
        question_char: (batch, seq_len, word_len)

    Outputs:
        ans_range_prop: (batch, 2, context_len)
        ans_range: (batch, 2)
        vis_alpha: to show on visdom
    """

    def __init__(self, global_config):
        super(MatchLSTMModel, self).__init__()

        # set config
        hidden_size = global_config['model']['global']['hidden_size']
        hidden_mode = global_config['model']['global']['hidden_mode']
        dropout_p = global_config['model']['global']['dropout_p']

        word_embedding_size = global_config['model']['encoder']['word_embedding_size']
        char_embedding_size = global_config['model']['encoder']['char_embedding_size']
        encoder_word_layers = global_config['model']['encoder']['word_layers']
        encoder_char_layers = global_config['model']['encoder']['char_layers']
        char_trainable = global_config['model']['encoder']['char_trainable']
        char_type = global_config['model']['encoder']['char_encode_type']
        char_cnn_filter_size = global_config['model']['encoder']['char_cnn_filter_size']
        char_cnn_filter_num = global_config['model']['encoder']['char_cnn_filter_num']
        self.enable_char = global_config['model']['encoder']['enable_char']

        # when mix-encode, use r-net methods, that concat char-encoding and word-embedding to represent sequence
        self.mix_encode = global_config['model']['encoder']['mix_encode']
        encoder_bidirection = global_config['model']['encoder']['bidirection']
        encoder_direction_num = 2 if encoder_bidirection else 1

        match_lstm_bidirection = global_config['model']['interaction']['match_lstm_bidirection']
        self_match_lstm_bidirection = global_config['model']['interaction']['self_match_bidirection']
        self.enable_self_match = global_config['model']['interaction']['enable_self_match']
        self.enable_birnn_after_self = global_config['model']['interaction']['birnn_after_self']
        gated_attention = global_config['model']['interaction']['gated_attention']
        mlp_attention = global_config['model']['interaction']['mlp_attention']
        self.enable_self_gated = global_config['model']['interaction']['self_gated']
        self.enable_question_match = global_config['model']['interaction']['question_match']

        match_rnn_direction_num = 2 if match_lstm_bidirection else 1
        self_match_rnn_direction_num = 2 if self_match_lstm_bidirection else 1

        ptr_bidirection = global_config['model']['output']['ptr_bidirection']
        ptr_direction_num = 2 if ptr_bidirection else 1
        self.init_ptr_hidden_mode = global_config['model']['output']['init_ptr_hidden']
        self.enable_search = global_config['model']['output']['answer_search']

        # construct model
        self.embedding = GloveEmbedding(dataset_h5_path=global_config['data']['dataset_h5'])
        encode_in_size = word_embedding_size

        if self.enable_char:
            self.char_embedding = CharEmbedding(dataset_h5_path=global_config['data']['dataset_h5'],
                                                embedding_size=char_embedding_size,
                                                trainable=char_trainable)
            if char_type == 'LSTM':
                self.char_encoder = CharEncoder(mode=hidden_mode,
                                                input_size=char_embedding_size,
                                                hidden_size=hidden_size,
                                                num_layers=encoder_char_layers,
                                                bidirectional=encoder_bidirection,
                                                dropout_p=dropout_p)
            elif char_type == 'CNN':
                self.char_encoder = CharCNNEncoder(emb_size=char_embedding_size,
                                                   hidden_size=word_embedding_size,
                                                   filters_size=char_cnn_filter_size,
                                                   filters_num=char_cnn_filter_num,
                                                   dropout_p=dropout_p)
            else:
                raise ValueError('Unrecognized char_encode_type of value %s' % char_type)
            if self.mix_encode:
                encode_in_size += hidden_size * encoder_direction_num

        self.encoder = MyRNNBase(mode=hidden_mode,
                                 input_size=encode_in_size,
                                 hidden_size=hidden_size,
                                 num_layers=encoder_word_layers,
                                 bidirectional=encoder_bidirection,
                                 dropout_p=dropout_p)
        encode_out_size = hidden_size * encoder_direction_num
        if self.enable_char and not self.mix_encode:
            encode_out_size *= 2

        match_rnn_in_size = encode_out_size
        if self.enable_question_match:
            self.question_match_rnn = MatchRNN(mode=hidden_mode,
                                               hp_input_size=encode_out_size,
                                               hq_input_size=encode_out_size,
                                               hidden_size=hidden_size,
                                               bidirectional=match_lstm_bidirection,
                                               gated_attention=gated_attention,
                                               mlp_attention=mlp_attention,
                                               dropout_p=dropout_p)
            match_rnn_in_size = hidden_size * match_rnn_direction_num

        self.match_rnn = MatchRNN(mode=hidden_mode,
                                  hp_input_size=encode_out_size,
                                  hq_input_size=match_rnn_in_size,
                                  hidden_size=hidden_size,
                                  bidirectional=match_lstm_bidirection,
                                  gated_attention=gated_attention,
                                  mlp_attention=mlp_attention,
                                  dropout_p=dropout_p)
        match_rnn_out_size = hidden_size * match_rnn_direction_num

        if self.enable_self_match:
            self.self_match_rnn = MatchRNN(mode=hidden_mode,
                                           hp_input_size=match_rnn_out_size,
                                           hq_input_size=match_rnn_out_size,
                                           hidden_size=hidden_size,
                                           bidirectional=self_match_lstm_bidirection,
                                           gated_attention=gated_attention,
                                           mlp_attention=mlp_attention,
                                           dropout_p=dropout_p)
            match_rnn_out_size = hidden_size * self_match_rnn_direction_num

        if self.enable_birnn_after_self:
            self.birnn_after_self = MyRNNBase(mode=hidden_mode,
                                              input_size=match_rnn_out_size,
                                              hidden_size=hidden_size,
                                              num_layers=1,
                                              bidirectional=True,
                                              dropout_p=dropout_p)
            match_rnn_out_size = hidden_size * 2

        if self.enable_self_gated:
            self.self_gated = SelfGated(input_size=match_rnn_out_size)

        self.pointer_net = BoundaryPointer(mode=hidden_mode,
                                           input_size=match_rnn_out_size,
                                           hidden_size=hidden_size,
                                           bidirectional=ptr_bidirection,
                                           dropout_p=dropout_p)
        ptr_in_size = hidden_size * ptr_direction_num

        # pointer net init hidden generate
        if self.init_ptr_hidden_mode == 'pooling':
            self.init_ptr_hidden = AttentionPooling(encode_out_size, ptr_in_size)
        elif self.init_ptr_hidden_mode == 'linear':
            self.init_ptr_hidden = nn.Linear(match_rnn_out_size, ptr_in_size)
        elif self.init_ptr_hidden_mode == 'None':
            pass
        else:
            raise ValueError('Wrong init_ptr_hidden mode select %s, change to pooling or linear'
                             % self.init_ptr_hidden_mode)

    def forward(self, context, question, context_char=None, question_char=None):
        if self.enable_char:
            assert context_char is not None and question_char is not None

        # get embedding: (seq_len, batch, embedding_size)
        context_vec, context_mask = self.embedding.forward(context)
        question_vec, question_mask = self.embedding.forward(question)

        # char-level embedding: (seq_len, batch, char_embedding_size)
        if self.enable_char:
            context_emb_char, context_char_mask = self.char_embedding.forward(context_char)
            question_emb_char, question_char_mask = self.char_embedding.forward(question_char)

            context_vec_char = self.char_encoder.forward(context_emb_char, context_char_mask, context_mask)
            question_vec_char = self.char_encoder.forward(question_emb_char, question_char_mask, question_mask)

            if self.mix_encode:
                context_vec = torch.cat((context_vec, context_vec_char), dim=-1)
                question_vec = torch.cat((question_vec, question_vec_char), dim=-1)

        # encode: (seq_len, batch, hidden_size)
        context_encode, _ = self.encoder.forward(context_vec, context_mask)
        question_encode, _ = self.encoder.forward(question_vec, question_mask)

        # char-level encode: (seq_len, batch, hidden_size)
        if self.enable_char and not self.mix_encode:
            context_encode = torch.cat((context_encode, context_vec_char), dim=-1)
            question_encode = torch.cat((question_encode, question_vec_char), dim=-1)

        # question match-lstm
        match_rnn_in_question = question_encode
        if self.enable_question_match:
            ct_aware_qt, _, _ = self.question_match_rnn.forward(question_encode, question_mask,
                                                                context_encode, context_mask)
            match_rnn_in_question = ct_aware_qt

        # match lstm: (seq_len, batch, hidden_size)
        qt_aware_ct, qt_aware_last_hidden, match_alpha = self.match_rnn.forward(context_encode, context_mask,
                                                                                match_rnn_in_question, question_mask)
        vis_param = {'match': match_alpha}

        # self match lstm: (seq_len, batch, hidden_size)
        if self.enable_self_match:
            qt_aware_ct, qt_aware_last_hidden, self_alpha = self.self_match_rnn.forward(qt_aware_ct, context_mask,
                                                                                        qt_aware_ct, context_mask)
            vis_param['self'] = self_alpha

        # birnn after self match: (seq_len, batch, hidden_size)
        if self.enable_birnn_after_self:
            qt_aware_ct, _ = self.birnn_after_self.forward(qt_aware_ct, context_mask)

        # self gated
        if self.enable_self_gated:
            qt_aware_ct = self.self_gated(qt_aware_ct)

        # pointer net init hidden: (batch, hidden_size)
        ptr_net_hidden = None
        if self.init_ptr_hidden_mode == 'pooling':
            ptr_net_hidden = self.init_ptr_hidden.forward(question_encode, question_mask)
        elif self.init_ptr_hidden_mode == 'linear':
            ptr_net_hidden = self.init_ptr_hidden.forward(qt_aware_last_hidden)
            ptr_net_hidden = F.tanh(ptr_net_hidden)

        # pointer net: (answer_len, batch, context_len)
        ans_range_prop = self.pointer_net.forward(qt_aware_ct, context_mask, ptr_net_hidden)
        ans_range_prop = ans_range_prop.transpose(0, 1)

        # answer range
        if self.enable_search:
            ans_range = answer_search(ans_range_prop, context_mask)
        else:
            ans_range = torch.max(ans_range_prop, 2)[1]

        return ans_range_prop, ans_range, vis_param
