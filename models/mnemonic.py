#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
from models.layers import *
from utils.functions import answer_search, multi_scale_ptr


class MReader(torch.nn.Module):
    """
    mnemonic reader model for machine comprehension
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

    def __init__(self, dataset_h5_path):
        super(MReader, self).__init__()

        # set config
        hidden_size = 100
        char_encoder_hidden = 50

        hidden_mode = 'LSTM'
        dropout_p = 0.2
        emb_dropout_p = 0.2
        enable_layer_norm = False

        word_embedding_size = 300   # manual set
        char_embedding_size = 50
        add_feature_size = 73   # manual set

        self.num_align_hops = 2
        self.num_ptr_hops = 2
        self.enable_search = True

        # construct model
        self.embedding = GloveEmbedding(dataset_h5_path=dataset_h5_path)
        self.char_embedding = CharEmbedding(dataset_h5_path=dataset_h5_path,
                                            embedding_size=char_embedding_size,
                                            trainable=True)

        self.char_encoder = CharEncoder(mode=hidden_mode,
                                        input_size=char_embedding_size,
                                        hidden_size=char_encoder_hidden,
                                        num_layers=1,
                                        bidirectional=True,
                                        dropout_p=emb_dropout_p)

        encoder_in_size = word_embedding_size + char_encoder_hidden * 2 + add_feature_size
        self.encoder = MyRNNBase(mode=hidden_mode,
                                 input_size=encoder_in_size,
                                 hidden_size=hidden_size,
                                 bidirectional=True,
                                 dropout_p=emb_dropout_p)

        self.aligner = torch.nn.ModuleList([SeqToSeqAtten() for _ in range(self.num_align_hops)])
        self.aligner_sfu = torch.nn.ModuleList([SFU(input_size=hidden_size * 2,
                                                    fusions_size=hidden_size * 2 * 3) for _ in
                                                range(self.num_align_hops)])
        self.self_aligner = torch.nn.ModuleList([SelfSeqAtten() for _ in range(self.num_align_hops)])
        self.self_aligner_sfu = torch.nn.ModuleList([SFU(input_size=hidden_size * 2,
                                                         fusions_size=hidden_size * 2 * 3)
                                                     for _ in range(self.num_align_hops)])
        self.aggregation = torch.nn.ModuleList([MyRNNBase(mode=hidden_mode,
                                                          input_size=hidden_size * 2,
                                                          hidden_size=hidden_size,
                                                          bidirectional=True,
                                                          dropout_p=dropout_p,
                                                          enable_layer_norm=enable_layer_norm)
                                                for _ in range(self.num_align_hops)])

        self.ptr_net = torch.nn.ModuleList([MemPtrNet(input_size=hidden_size * 2,
                                                      hidden_size=hidden_size,
                                                      dropout_p=dropout_p) for _ in range(self.num_ptr_hops)])

    def forward(self, context, question, context_char=None, question_char=None, context_f=None, question_f=None):
        assert context_char is not None and question_char is not None and context_f is not None \
               and question_f is not None

        vis_param = {}

        # (seq_len, batch, additional_feature_size)
        context_f = context_f.transpose(0, 1)
        question_f = question_f.transpose(0, 1)

        # word-level embedding: (seq_len, batch, word_embedding_size)
        context_vec, context_mask = self.embedding.forward(context)
        question_vec, question_mask = self.embedding.forward(question)

        # char-level embedding: (seq_len, batch, char_embedding_size)
        context_emb_char, context_char_mask = self.char_embedding.forward(context_char)
        question_emb_char, question_char_mask = self.char_embedding.forward(question_char)

        context_vec_char = self.char_encoder.forward(context_emb_char, context_char_mask, context_mask)
        question_vec_char = self.char_encoder.forward(question_emb_char, question_char_mask, question_mask)

        # mix embedding: (seq_len, batch, embedding_size)
        context_vec = torch.cat((context_vec, context_vec_char, context_f), dim=-1)
        question_vec = torch.cat((question_vec, question_vec_char, question_f), dim=-1)

        # encode: (seq_len, batch, hidden_size*2)
        context_encode, _ = self.encoder.forward(context_vec, context_mask)
        question_encode, zs = self.encoder.forward(question_vec, question_mask)

        align_ct = context_encode
        for i in range(self.num_align_hops):
            # align: (seq_len, batch, hidden_size*2)
            qt_align_ct, alpha = self.aligner[i](align_ct, question_encode, question_mask)
            bar_ct = self.aligner_sfu[i](align_ct, torch.cat([qt_align_ct,
                                                              align_ct * qt_align_ct,
                                                              align_ct - qt_align_ct], dim=-1))
            vis_param['match'] = alpha

            # self-align: (seq_len, batch, hidden_size*2)
            ct_align_ct, self_alpha = self.self_aligner[i](bar_ct, context_mask)
            hat_ct = self.self_aligner_sfu[i](bar_ct, torch.cat([ct_align_ct,
                                                                 bar_ct * ct_align_ct,
                                                                 bar_ct - ct_align_ct], dim=-1))
            vis_param['self-match'] = self_alpha

            # aggregation: (seq_len, batch, hidden_size*2)
            align_ct, _ = self.aggregation[i](hat_ct, context_mask)

        # pointer net: (answer_len, batch, context_len)
        for i in range(self.num_ptr_hops):
            ans_range_prop, zs = self.ptr_net[i](align_ct, context_mask, zs)

        # answer range
        ans_range_prop = ans_range_prop.transpose(0, 1)
        if not self.training and self.enable_search:
            ans_range = answer_search(ans_range_prop, context_mask)
        else:
            _, ans_range = torch.max(ans_range_prop, dim=2)

        return ans_range_prop, ans_range, vis_param
