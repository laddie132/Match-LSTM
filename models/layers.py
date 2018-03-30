#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import h5py
import torch
import torch.nn.functional as F
import numpy as np
from dataset.preprocess_data import PreprocessData
from utils.utils import masked_softmax, generate_mask, compute_mask


class GloveEmbedding(torch.nn.Module):
    """
    Glove Embedding Layer, also compute the mask of padding index
    Args:
        - glove_h5_path: glove embedding file path
        - dropout_p: dropout probability
    Inputs:
        **input** (batch, seq_len): sequence with word index
    Outputs
        **output** (batch, seq_len, embedding_size): tensor that change word index to word embeddings
        **mask** (batch, seq_len): tensor that show which index is padding
    """

    def __init__(self, dataset_h5_path, dropout_p=0.):
        super(GloveEmbedding, self).__init__()
        self.dataset_h5_path = dataset_h5_path
        self.n_embeddings, self.len_embedding, self.weights = self.load_glove_hdf5()

        self.embedding_layer = torch.nn.Embedding(num_embeddings=self.n_embeddings, embedding_dim=self.len_embedding)
        self.embedding_layer.weight = torch.nn.Parameter(self.weights)
        self.embedding_layer.weight.requires_grad = False

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def load_glove_hdf5(self):
        with h5py.File(self.dataset_h5_path, 'r') as f:
            f_meta_data = f['meta_data']
            id2vec = np.array(f_meta_data['id2vec'])  # only need 1.11s
            word_dict_size = f.attrs['word_dict_size']
            embedding_size = f.attrs['embedding_size']

        return int(word_dict_size), int(embedding_size), torch.from_numpy(id2vec)

    def forward(self, x):
        mask = compute_mask(x, PreprocessData.padding_idx)

        tmp_emb = self.embedding_layer.forward(x)
        out_emb = self.dropout(tmp_emb)

        return out_emb, mask


class MatchLSTMAttention(torch.nn.Module):
    r"""
    attention mechanism in match-lstm
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr

    Inputs:
        Hp(1, batch, input_size): a context word encoded
        Hq(question_len, batch, input_size): whole question encoded
        Hr_last(batch, hidden_size): last lstm hidden output

    Outputs:
        alpha(batch, question_len): attention vector
    """

    def __init__(self, input_size, hidden_size):
        super(MatchLSTMAttention, self).__init__()

        self.linear_wq = torch.nn.Linear(input_size, hidden_size)
        self.linear_wp = torch.nn.Linear(input_size, hidden_size)
        self.linear_wr = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_wg = torch.nn.Linear(hidden_size, 1)

    def forward(self, Hpi, Hq, Hr_last, Hq_mask):
        wq_hq = self.linear_wq(Hq)  # (question_len, batch, hidden_size)
        wp_hp = self.linear_wp(Hpi).unsqueeze(0)  # (1, batch, hidden_size)
        wr_hr = self.linear_wr(Hr_last).unsqueeze(0)  # (1, batch, hidden_size)
        G = F.tanh(wq_hq + wp_hp + wr_hr)  # (question_len, batch, hidden_size), auto broadcast
        wg_g = self.linear_wg(G) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, question_len)
        alpha = masked_softmax(wg_g, m=Hq_mask, dim=1)  # (batch, question_len)
        return alpha


class UniMatchLSTM(torch.nn.Module):
    r"""
    interaction context and question with attention mechanism, one direction
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr

    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded

    Outputs:
        Hr(context_len, batch, hidden_size): question-aware context representation
    """

    def __init__(self, input_size, hidden_size):
        super(UniMatchLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.attention = MatchLSTMAttention(input_size, hidden_size)
        self.lstm = torch.nn.LSTMCell(input_size=2 * input_size, hidden_size=hidden_size)

    def forward(self, Hp, Hq, Hq_mask):
        batch_size = Hp.shape[1]
        context_len = Hp.shape[0]

        # init hidden with the same type of input data
        h_0 = torch.autograd.Variable(Hq.data.new(batch_size, self.hidden_size).zero_())
        hidden = [(h_0, h_0)]

        for t in range(context_len):
            cur_hp = Hp[t, ...]  # (batch, input_size)
            alpha = self.attention.forward(cur_hp, Hq, hidden[t][0], Hq_mask)  # (batch, question_len)
            question_alpha = torch.bmm(alpha.unsqueeze(1), Hq.transpose(0, 1)) \
                .squeeze(1)  # (batch, input_size)
            cur_z = torch.cat([cur_hp, question_alpha], dim=1)  # (batch, 2*input_size)

            cur_hidden = self.lstm.forward(cur_z, hidden[t])  # (batch, hidden_size), (batch, hidden_size)
            hidden.append(cur_hidden)

        hidden_state = map(lambda x: x[0], hidden)
        result = torch.stack(list(hidden_state)[1:], dim=0)  # (context_len, batch, hidden_size)
        return result


class MatchLSTM(torch.nn.Module):
    r"""
    interaction context and question with attention mechanism
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr
        - bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        - enable_cuda: enable GPU accelerate or not

    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded
        Hp_length(batch,): each context valued length without padding values
        Hq_length(batch,): each question valued length without padding values

    Outputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
    """

    def __init__(self, input_size, hidden_size, bidirectional, enable_cuda):
        super(MatchLSTM, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 1 if bidirectional else 2
        self.enable_cuda = enable_cuda

        self.left_match_lstm = UniMatchLSTM(input_size, hidden_size)

        if bidirectional:
            self.right_match_lstm = UniMatchLSTM(input_size, hidden_size)

    def flip(self, vin, mask):
        """
        flip a tensor
        :param vin: input batch with padding values
        :param mask: show whether padding index
        :return:
        """
        length = mask.data.eq(1).long().sum(1).squeeze()        # todo: speed up, vectoration

        flip_list = []
        for i in range(vin.shape[1]):
            cur_tensor = vin[:, i, :]
            cur_length = length[i]

            idx = list(range(cur_length - 1, -1, -1)) + list(range(cur_length, cur_tensor.shape[0]))
            idx = torch.autograd.Variable(torch.LongTensor(idx))
            if self.enable_cuda:
                idx = idx.cuda()

            cur_inv_tensor = cur_tensor.index_select(0, idx)
            flip_list.append(cur_inv_tensor)
        inv_tensor = torch.stack(flip_list, dim=1)

        return inv_tensor

    def forward(self, Hp, Hp_mask, Hq, Hq_mask):
        left_hidden = self.left_match_lstm.forward(Hp, Hq, Hq_mask)
        rtn_hidden = left_hidden

        if self.bidirectional:
            Hp_inv = self.flip(Hp, Hp_mask)
            right_hidden = self.right_match_lstm.forward(Hp_inv, Hq, Hq_mask)
            rtn_hidden = torch.cat((left_hidden, right_hidden), dim=2)

        real_rtn_hidden = Hp_mask.transpose(0, 1).unsqueeze(2) * rtn_hidden
        last_hidden = rtn_hidden[-1, :]

        return real_rtn_hidden, last_hidden


class PointerAttention(torch.nn.Module):
    r"""
    attention mechanism in pointer network
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer

    Inputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
        Hk_last(batch, hidden_size): the last hidden output of previous time

    Outputs:
        beta(batch, context_len): question-aware context representation
    """

    def __init__(self, input_size, hidden_size):
        super(PointerAttention, self).__init__()

        self.linear_wr = torch.nn.Linear(input_size, hidden_size)
        self.linear_wa = torch.nn.Linear(hidden_size, hidden_size)
        self.linear_wf = torch.nn.Linear(hidden_size, 1)

    def forward(self, Hr, Hr_mask, Hk_pre):
        wr_hr = self.linear_wr(Hr)  # (context_len, batch, hidden_size)
        wa_ha = self.linear_wa(Hk_pre).unsqueeze(0)  # (1, batch, hidden_size)
        f = F.tanh(wr_hr + wa_ha)  # (context_len, batch, hidden_size)

        beta_tmp = self.linear_wf(f) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, context_len)

        beta = masked_softmax(beta_tmp, m=Hr_mask, dim=1)
        return beta


class SeqPointer(torch.nn.Module):
    r"""
    Sequence Pointer Net that output every possible answer position in context
    Args:

    Inputs:
        Hr: question-aware context representation
    Outputs:
        **output** every answer index possibility position in context, no fixed length
    """

    def __init__(self):
        super(SeqPointer, self).__init__()

    def forward(self, *input):
        pass


class BoundaryPointer(torch.nn.Module):
    r"""
    boundary Pointer Net that output start and end possible answer position in context
    Args:
        - input_size: The number of features in Hr
        - hidden_size: The number of features in the hidden layer

    Inputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
        h_0(batch, hidden_size): init lstm cell hidden state
    Outputs:
        **output** (answer_len, batch, context_len): start and end answer index possibility position in context
    """
    answer_len = 2

    def __init__(self, input_size, hidden_size, dropout_p):
        super(BoundaryPointer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.attention = PointerAttention(input_size, hidden_size)
        self.lstm = torch.nn.LSTMCell(input_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hr, Hr_mask, h_0=None):
        Hr = self.dropout.forward(Hr)

        if h_0 is None:
            batch_size = Hr.shape[1]
            h_0 = torch.autograd.Variable(Hr.data.new(batch_size, self.hidden_size).zero_())
        hidden = (h_0, h_0)
        beta_out = []

        for t in range(self.answer_len):
            beta = self.attention.forward(Hr, Hr_mask, hidden[0])  # (batch, context_len)
            beta_out.append(beta)

            context_beta = torch.bmm(beta.unsqueeze(1), Hr.transpose(0, 1)) \
                .squeeze(1)  # (batch, input_size)

            hidden = self.lstm.forward(context_beta, hidden)  # (batch, hidden_size), (batch, hidden_size)

        result = torch.stack(beta_out, dim=0)

        # todo: unexplainable
        new_mask = torch.neg((Hr_mask - 1) * 1e-6)    # mask replace zeros with 1e-6, make sure no gradient explosion
        result = result + new_mask.unsqueeze(0)

        return result


class MyLSTM(torch.nn.Module):
    """
    LSTM with packed sequence and dropout
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        dropout_p: dropout probability to input data

    Inputs: input, mask
        - **input** (batch, seq_len, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output, output_mask
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t.
        - **output_length** (batch, seq_len): list of each sample true length

    """

    def __init__(self, input_size, hidden_size, bidirectional, dropout_p):
        super(MyLSTM, self).__init__()

        self.lstm = torch.nn.LSTM(input_size=input_size,
                                  hidden_size=hidden_size,
                                  bidirectional=bidirectional)
        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, v, mask):
        # get sorted v
        lengths = mask.data.eq(1).long().sum(1).squeeze()
        _, idx_sort = torch.sort(lengths, dim=0, descending=True)
        _, idx_unsort = torch.sort(idx_sort, dim=0)

        lengths_sort = list(lengths[idx_sort])
        idx_sort = torch.autograd.Variable(idx_sort)
        idx_unsort = torch.autograd.Variable(idx_unsort)

        if v.is_cuda:
            idx_sort = idx_sort.cuda()
            idx_unsort = idx_unsort.cuda()

        v_sort = v.index_select(0, idx_sort)
        v_sort = v_sort.transpose(0, 1)

        v_pack = torch.nn.utils.rnn.pack_padded_sequence(v_sort, lengths_sort)
        v_dropout = self.dropout.forward(v_pack.data)
        v_pack_dropout = torch.nn.utils.rnn.PackedSequence(v_dropout, v_pack.batch_sizes)

        o_pack_dropout, _ = self.lstm.forward(v_pack_dropout)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)

        # unsorted o
        o_unsort = o.index_select(1, idx_unsort)        # notice here first dim is seq_len
        new_mask = generate_mask(lengths_sort, enable_cuda=v.is_cuda)
        new_mask_unsort = new_mask.index_select(0, idx_unsort)

        return o_unsort, new_mask_unsort
