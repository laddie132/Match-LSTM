#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
import torch.nn as nn
from models.layers import *
from utils.functions import answer_search, multi_scale_ptr


class BaseModel(torch.nn.Module):
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

    def __init__(self, dataset_h5_path, model_config):
        super(BaseModel, self).__init__()

        # set config
        hidden_size = model_config['global']['hidden_size']
        hidden_mode = model_config['global']['hidden_mode']
        dropout_p = model_config['global']['dropout_p']
        emb_dropout_p = model_config['global']['emb_dropout_p']
        enable_layer_norm = model_config['global']['layer_norm']

        word_embedding_size = model_config['encoder']['word_embedding_size']
        char_embedding_size = model_config['encoder']['char_embedding_size']
        encoder_word_layers = model_config['encoder']['word_layers']
        encoder_char_layers = model_config['encoder']['char_layers']
        char_trainable = model_config['encoder']['char_trainable']
        char_type = model_config['encoder']['char_encode_type']
        char_cnn_filter_size = model_config['encoder']['char_cnn_filter_size']
        char_cnn_filter_num = model_config['encoder']['char_cnn_filter_num']
        self.enable_char = model_config['encoder']['enable_char']
        add_features = model_config['encoder']['add_features']
        self.enable_features = True if add_features > 0 else False

        # when mix-encode, use r-net methods, that concat char-encoding and word-embedding to represent sequence
        self.mix_encode = model_config['encoder']['mix_encode']
        encoder_bidirection = model_config['encoder']['bidirection']
        encoder_direction_num = 2 if encoder_bidirection else 1

        match_lstm_bidirection = model_config['interaction']['match_lstm_bidirection']
        self_match_lstm_bidirection = model_config['interaction']['self_match_bidirection']
        self.enable_self_match = model_config['interaction']['enable_self_match']
        self.enable_birnn_after_self = model_config['interaction']['birnn_after_self']
        gated_attention = model_config['interaction']['gated_attention']
        self.enable_self_gated = model_config['interaction']['self_gated']
        self.enable_question_match = model_config['interaction']['question_match']

        match_rnn_direction_num = 2 if match_lstm_bidirection else 1
        self_match_rnn_direction_num = 2 if self_match_lstm_bidirection else 1

        num_hops = model_config['output']['num_hops']
        self.scales = model_config['output']['scales']
        ptr_bidirection = model_config['output']['ptr_bidirection']
        self.init_ptr_hidden_mode = model_config['output']['init_ptr_hidden']
        self.enable_search = model_config['output']['answer_search']

        assert num_hops > 0, 'Pointer Net number of hops should bigger than zero'
        if num_hops > 1:
            assert not ptr_bidirection, 'Pointer Net bidirectional should with number of one hop'

        # construct model
        self.embedding = GloveEmbedding(dataset_h5_path=dataset_h5_path)
        encode_in_size = word_embedding_size + add_features

        if self.enable_char:
            self.char_embedding = CharEmbedding(dataset_h5_path=dataset_h5_path,
                                                embedding_size=char_embedding_size,
                                                trainable=char_trainable)
            if char_type == 'LSTM':
                self.char_encoder = CharEncoder(mode=hidden_mode,
                                                input_size=char_embedding_size,
                                                hidden_size=hidden_size,
                                                num_layers=encoder_char_layers,
                                                bidirectional=encoder_bidirection,
                                                dropout_p=emb_dropout_p)
            elif char_type == 'CNN':
                self.char_encoder = CharCNNEncoder(emb_size=char_embedding_size,
                                                   hidden_size=hidden_size,
                                                   filters_size=char_cnn_filter_size,
                                                   filters_num=char_cnn_filter_num,
                                                   dropout_p=emb_dropout_p)
            else:
                raise ValueError('Unrecognized char_encode_type of value %s' % char_type)
            if self.mix_encode:
                encode_in_size += hidden_size * encoder_direction_num

        self.encoder = MyStackedRNN(mode=hidden_mode,
                                    input_size=encode_in_size,
                                    hidden_size=hidden_size,
                                    num_layers=encoder_word_layers,
                                    bidirectional=encoder_bidirection,
                                    dropout_p=emb_dropout_p)
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
                                               dropout_p=dropout_p,
                                               enable_layer_norm=enable_layer_norm)
            match_rnn_in_size = hidden_size * match_rnn_direction_num

        self.match_rnn = MatchRNN(mode=hidden_mode,
                                  hp_input_size=encode_out_size,
                                  hq_input_size=match_rnn_in_size,
                                  hidden_size=hidden_size,
                                  bidirectional=match_lstm_bidirection,
                                  gated_attention=gated_attention,
                                  dropout_p=dropout_p,
                                  enable_layer_norm=enable_layer_norm)
        match_rnn_out_size = hidden_size * match_rnn_direction_num

        if self.enable_self_match:
            self.self_match_rnn = MatchRNN(mode=hidden_mode,
                                           hp_input_size=match_rnn_out_size,
                                           hq_input_size=match_rnn_out_size,
                                           hidden_size=hidden_size,
                                           bidirectional=self_match_lstm_bidirection,
                                           gated_attention=gated_attention,
                                           dropout_p=dropout_p,
                                           enable_layer_norm=enable_layer_norm)
            match_rnn_out_size = hidden_size * self_match_rnn_direction_num

        if self.enable_birnn_after_self:
            self.birnn_after_self = MyRNNBase(mode=hidden_mode,
                                              input_size=match_rnn_out_size,
                                              hidden_size=hidden_size,
                                              bidirectional=True,
                                              dropout_p=dropout_p,
                                              enable_layer_norm=enable_layer_norm)
            match_rnn_out_size = hidden_size * 2

        if self.enable_self_gated:
            self.self_gated = SelfGated(input_size=match_rnn_out_size)

        if num_hops == 1:
            self.pointer_net = torch.nn.ModuleList([BoundaryPointer(mode=hidden_mode,
                                                                    input_size=match_rnn_out_size,
                                                                    hidden_size=hidden_size,
                                                                    bidirectional=ptr_bidirection,
                                                                    dropout_p=dropout_p,
                                                                    enable_layer_norm=enable_layer_norm) for _ in
                                                    range(len(self.scales))])
        else:
            self.pointer_net = torch.nn.ModuleList([MultiHopBdPointer(mode=hidden_mode,
                                                                      input_size=match_rnn_out_size,
                                                                      hidden_size=hidden_size,
                                                                      num_hops=num_hops,
                                                                      dropout_p=dropout_p,
                                                                      enable_layer_norm=enable_layer_norm) for _ in
                                                    range(len(self.scales))])

        # pointer net init hidden generate
        if self.init_ptr_hidden_mode == 'pooling':
            self.init_ptr_hidden = AttentionPooling(encode_out_size, hidden_size)
        elif self.init_ptr_hidden_mode == 'linear':
            self.init_ptr_hidden = nn.Linear(match_rnn_out_size, hidden_size)
        elif self.init_ptr_hidden_mode == 'None':
            pass
        else:
            raise ValueError('Wrong init_ptr_hidden mode select %s, change to pooling or linear'
                             % self.init_ptr_hidden_mode)

    def forward(self, context, question, context_char=None, question_char=None, context_f=None, question_f=None):
        if self.enable_char:
            assert context_char is not None and question_char is not None

        if self.enable_features:
            assert context_f is not None and question_f is not None

        # get embedding: (seq_len, batch, embedding_size)
        context_vec, context_mask = self.embedding.forward(context)
        question_vec, question_mask = self.embedding.forward(question)

        if self.enable_features:
            assert context_f is not None and question_f is not None

            # (seq_len, batch, additional_feature_size)
            context_f = context_f.transpose(0, 1)
            question_f = question_f.transpose(0, 1)

            context_vec = torch.cat([context_vec, context_f], dim=-1)
            question_vec = torch.cat([question_vec, question_f], dim=-1)

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
        qt_aware_ct, qt_aware_last_hidden, match_para = self.match_rnn.forward(context_encode, context_mask,
                                                                               match_rnn_in_question, question_mask)
        vis_param = {'match': match_para}

        # self match lstm: (seq_len, batch, hidden_size)
        if self.enable_self_match:
            qt_aware_ct, qt_aware_last_hidden, self_para = self.self_match_rnn.forward(qt_aware_ct, context_mask,
                                                                                       qt_aware_ct, context_mask)
            vis_param['self'] = self_para

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
        # ans_range_prop = self.pointer_net.forward(qt_aware_ct, context_mask, ptr_net_hidden)
        # ans_range_prop = ans_range_prop.transpose(0, 1)

        ans_range_prop = multi_scale_ptr(self.pointer_net, ptr_net_hidden, qt_aware_ct, context_mask, self.scales)

        # answer range
        if not self.training and self.enable_search:
            ans_range = answer_search(ans_range_prop, context_mask)
        else:
            _, ans_range = torch.max(ans_range_prop, dim=2)

        return ans_range_prop, ans_range, vis_param
