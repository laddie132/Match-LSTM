#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import math
import h5py
import torch
import torch.nn.functional as F
import numpy as np
from dataset.preprocess_data import PreprocessData
from utils.functions import masked_softmax, compute_mask, masked_flip


class GloveEmbedding(torch.nn.Module):
    """
    Glove Embedding Layer, also compute the mask of padding index
    Args:
        - dataset_h5_path: glove embedding file path
    Inputs:
        **input** (batch, seq_len): sequence with word index
    Outputs
        **output** (seq_len, batch, embedding_size): tensor that change word index to word embeddings
        **mask** (batch, seq_len): tensor that show which index is padding
    """

    def __init__(self, dataset_h5_path):
        super(GloveEmbedding, self).__init__()
        self.dataset_h5_path = dataset_h5_path
        n_embeddings, len_embedding, weights = self.load_glove_hdf5()

        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_embeddings, embedding_dim=len_embedding)
        self.embedding_layer.weight = torch.nn.Parameter(weights)
        self.embedding_layer.weight.requires_grad = False

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
        out_emb = tmp_emb.transpose(0, 1)

        return out_emb, mask


class CharEmbedding(torch.nn.Module):
    """
    Char Embedding Layer, random weight
    Args:
        - dataset_h5_path: char embedding file path
    Inputs:
        **input** (batch, seq_len, word_len): word sequence with char index
    Outputs
        **output** (batch, seq_len, word_len, embedding_size): tensor contain char embeddings
        **mask** (batch, seq_len, word_len)
    """

    def __init__(self, dataset_h5_path, embedding_size, trainable=False):
        super(CharEmbedding, self).__init__()
        self.dataset_h5_path = dataset_h5_path
        n_embedding = self.load_dataset_h5()

        self.embedding_layer = torch.nn.Embedding(num_embeddings=n_embedding, embedding_dim=embedding_size,
                                                  padding_idx=0)

        # notice that cannot directly assign value. When in predict, it's always False.
        if not trainable:
            self.embedding_layer.weight.requires_grad = False

    def load_dataset_h5(self):
        with h5py.File(self.dataset_h5_path, 'r') as f:
            word_dict_size = f.attrs['char_dict_size']

        return int(word_dict_size)

    def forward(self, x):
        batch_size, seq_len, word_len = x.shape
        x = x.view(-1, word_len)

        mask = compute_mask(x, 0)  # char-level padding idx is zero
        x_emb = self.embedding_layer.forward(x)
        x_emb = x_emb.view(batch_size, seq_len, word_len, -1)
        mask = mask.view(batch_size, seq_len, word_len)

        return x_emb, mask


class CharEncoder(torch.nn.Module):
    """
    char-level encoder with MyRNNBase used
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, mode, input_size, hidden_size, num_layers, bidirectional, dropout_p):
        super(CharEncoder, self).__init__()

        self.encoder = MyRNNBase(mode=mode,
                                 input_size=input_size,
                                 hidden_size=hidden_size,
                                 num_layers=num_layers,
                                 bidirectional=bidirectional,
                                 dropout_p=dropout_p)

    def forward(self, x, char_mask, word_mask):
        batch_size, seq_len, word_len, embedding_size = x.shape
        x = x.view(-1, word_len, embedding_size)
        x = x.transpose(0, 1)
        char_mask = char_mask.view(-1, word_len)

        _, x_encode = self.encoder.forward(x, char_mask)  # (batch*seq_len, hidden_size)
        x_encode = x_encode.view(batch_size, seq_len, -1)  # (batch, seq_len, hidden_size)
        x_encode = x_encode * word_mask.unsqueeze(-1)

        return x_encode.transpose(0, 1)


class CharCNN(torch.nn.Module):
    """
    Char-level CNN
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """

    def __init__(self, emb_size, filters_size, filters_num, dropout_p):
        super(CharCNN, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.cnns = torch.nn.ModuleList(
            [torch.nn.Conv2d(1, fn, (fw, emb_size)) for fw, fn in zip(filters_size, filters_num)])

    def forward(self, x, char_mask, word_mask):
        x = self.dropout(x)

        batch_size, seq_len, word_len, embedding_size = x.shape
        x = x.view(-1, word_len, embedding_size).unsqueeze(1)  # (N, 1, word_len, embedding_size)

        x = [F.relu(cnn(x)).squeeze(-1) for cnn in self.cnns]  # (N, Cout, word_len - fw + 1) * fn
        x = [torch.max(cx, 2)[0] for cx in x]  # (N, Cout) * fn
        x = torch.cat(x, dim=1)  # (N, hidden_size)

        x = x.view(batch_size, seq_len, -1)  # (batch, seq_len, hidden_size)
        x = x * word_mask.unsqueeze(-1)

        return x.transpose(0, 1)


class Highway(torch.nn.Module):
    def __init__(self, in_size, n_layers, dropout_p):
        super(Highway, self).__init__()
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.normal_layer = torch.nn.ModuleList([torch.nn.Linear(in_size, in_size) for _ in range(n_layers)])
        self.gate_layer = torch.nn.ModuleList([torch.nn.Linear(in_size, in_size) for _ in range(n_layers)])

    def forward(self, x):
        for i in range(self.n_layers):
            x = F.dropout(x, p=self.dropout_p, training=self.training)

            normal_layer_ret = F.relu(self.normal_layer[i](x))
            gate = F.sigmoid(self.gate_layer[i](x))

            x = gate * normal_layer_ret + (1 - gate) * x
        return x


class CharCNNEncoder(torch.nn.Module):
    """
    char-level cnn encoder with highway networks
    Inputs:
        **input** (batch, seq_len, word_len, embedding_size)
        **char_mask** (batch, seq_len, word_len)
        **word_mask** (batch, seq_len)
    Outputs
        **output** (seq_len, batch, hidden_size)
    """
    def __init__(self, emb_size, hidden_size, filters_size, filters_num, dropout_p, enable_highway=True):
        super(CharCNNEncoder, self).__init__()
        self.enable_highway = enable_highway
        self.hidden_size = hidden_size

        self.cnn = CharCNN(emb_size=emb_size,
                           filters_size=filters_size,
                           filters_num=filters_num,
                           dropout_p=dropout_p)

        if enable_highway:
            self.highway = Highway(in_size=hidden_size,
                                   n_layers=2,
                                   dropout_p=dropout_p)

    def forward(self, x, char_mask, word_mask):
        o = self.cnn(x, char_mask, word_mask)

        assert o.shape[2] == self.hidden_size
        if self.enable_highway:
            o = self.highway(o)

        return o


class MatchRNNAttention(torch.nn.Module):
    r"""
    attention mechanism in match-rnn
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr

    Inputs:
        Hpi(batch, input_size): a context word encoded
        Hq(question_len, batch, input_size): whole question encoded
        Hr_last(batch, hidden_size): last lstm hidden output

    Outputs:
        alpha(batch, question_len): attention vector
    """

    def __init__(self, input_size, hidden_size):
        super(MatchRNNAttention, self).__init__()

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


class UniMatchRNN(torch.nn.Module):
    r"""
    interaction context and question with attention mechanism, one direction, using LSTM cell
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr

    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded

    Outputs:
        Hr(context_len, batch, hidden_size): question-aware context representation
        alpha(batch, question_len, context_len): used for visual show
    """

    def __init__(self, mode, input_size, hidden_size, gated_attention=False):
        super(UniMatchRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.attention = MatchRNNAttention(input_size, hidden_size)

        self.gated_attention = gated_attention

        if self.gated_attention:
            self.gated_linear = torch.nn.Linear(2 * input_size, 2 * input_size)

        self.mode = mode
        if mode == 'LSTM':
            self.hidden_cell = torch.nn.LSTMCell(input_size=2 * input_size, hidden_size=hidden_size)
        elif mode == 'GRU':
            self.hidden_cell = torch.nn.GRUCell(input_size=2 * input_size, hidden_size=hidden_size)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform(t)
        for t in hh:
            torch.nn.init.orthogonal(t)
        for t in b:
            torch.nn.init.constant(t, 0)

    def forward(self, Hp, Hq, Hq_mask):
        batch_size = Hp.shape[1]
        context_len = Hp.shape[0]

        # init hidden with the same type of input data
        h_0 = torch.autograd.Variable(Hq.data.new(batch_size, self.hidden_size).zero_())
        hidden = [(h_0, h_0)] if self.mode == 'LSTM' else [h_0]
        vis_alpha = []

        for t in range(context_len):
            cur_hp = Hp[t, ...]  # (batch, input_size)
            attention_input = hidden[t][0] if self.mode == 'LSTM' else hidden[t]

            alpha = self.attention.forward(cur_hp, Hq, attention_input, Hq_mask)  # (batch, question_len)
            vis_alpha.append(alpha)

            question_alpha = torch.bmm(alpha.unsqueeze(1), Hq.transpose(0, 1)) \
                .squeeze(1)  # (batch, input_size)
            cur_z = torch.cat([cur_hp, question_alpha], dim=1)  # (batch, 2*input_size)

            # gated
            if self.gated_attention:
                cur_z = F.sigmoid(self.gated_linear.forward(cur_z))

            cur_hidden = self.hidden_cell.forward(cur_z, hidden[t])  # (batch, hidden_size), when lstm output tuple
            hidden.append(cur_hidden)

        vis_alpha = torch.stack(vis_alpha, dim=2)  # (batch, question_len, context_len)

        hidden_state = list(map(lambda x: x[0], hidden)) if self.mode == 'LSTM' else hidden
        result = torch.stack(hidden_state[1:], dim=0)  # (context_len, batch, hidden_size)
        return result, vis_alpha


class MatchRNN(torch.nn.Module):
    r"""
    interaction context and question with attention mechanism
    Args:
        - input_size: The number of expected features in the input Hp and Hq
        - hidden_size: The number of features in the hidden state Hr
        - bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        - gated_attention: If ``True``, gated attention used, see more on R-NET

    Inputs:
        Hp(context_len, batch, input_size): context encoded
        Hq(question_len, batch, input_size): question encoded
        Hp_mask(batch, context_len): each context valued length without padding values
        Hq_mask(batch, question_len): each question valued length without padding values

    Outputs:
        Hr(context_len, batch, hidden_size * num_directions): question-aware context representation
    """

    def __init__(self, mode, input_size, hidden_size, bidirectional, gated_attention, dropout_p):
        super(MatchRNN, self).__init__()
        self.bidirectional = bidirectional
        self.num_directions = 1 if bidirectional else 2

        self.left_match_rnn = UniMatchRNN(mode, input_size, hidden_size, gated_attention)
        if bidirectional:
            self.right_match_rnn = UniMatchRNN(mode, input_size, hidden_size, gated_attention)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hp, Hp_mask, Hq, Hq_mask):
        Hp = self.dropout(Hp)  # todo: move it to rnn of each direction
        Hq = self.dropout(Hq)

        left_hidden, left_alpha = self.left_match_rnn.forward(Hp, Hq, Hq_mask)
        rtn_hidden = left_hidden
        rtn_alpha = {'left': left_alpha}

        if self.bidirectional:
            Hp_inv = masked_flip(Hp, Hp_mask, flip_dim=0)
            right_hidden, right_alpha = self.right_match_rnn.forward(Hp_inv, Hq, Hq_mask)
            rtn_hidden = torch.cat((left_hidden, right_hidden), dim=2)
            rtn_alpha['right'] = right_alpha

        real_rtn_hidden = Hp_mask.transpose(0, 1).unsqueeze(2) * rtn_hidden
        last_hidden = rtn_hidden[-1, :]

        return real_rtn_hidden, last_hidden, rtn_alpha


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
        return NotImplementedError


class UniBoundaryPointer(torch.nn.Module):
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

    def __init__(self, mode, input_size, hidden_size):
        super(UniBoundaryPointer, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.attention = PointerAttention(input_size, hidden_size)

        self.mode = mode
        if mode == 'LSTM':
            self.hidden_cell = torch.nn.LSTMCell(input_size, hidden_size)
        elif mode == 'GRU':
            self.hidden_cell = torch.nn.GRUCell(input_size, hidden_size)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform(t)
        for t in hh:
            torch.nn.init.orthogonal(t)
        for t in b:
            torch.nn.init.constant(t, 0)

    def forward(self, Hr, Hr_mask, h_0=None):
        if h_0 is None:
            batch_size = Hr.shape[1]
            h_0 = torch.autograd.Variable(Hr.data.new(batch_size, self.hidden_size).zero_())

        hidden = (h_0, h_0) if self.mode == 'LSTM' else h_0
        beta_out = []

        for t in range(self.answer_len):
            attention_input = hidden[0] if self.mode == 'LSTM' else hidden
            beta = self.attention.forward(Hr, Hr_mask, attention_input)  # (batch, context_len)
            beta_out.append(beta)

            context_beta = torch.bmm(beta.unsqueeze(1), Hr.transpose(0, 1)) \
                .squeeze(1)  # (batch, input_size)

            hidden = self.hidden_cell.forward(context_beta, hidden)  # (batch, hidden_size), (batch, hidden_size)

        result = torch.stack(beta_out, dim=0)
        return result


class BoundaryPointer(torch.nn.Module):

    def __init__(self, mode, input_size, hidden_size, bidirectional, dropout_p):
        super(BoundaryPointer, self).__init__()
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size

        self.left_ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size)
        if bidirectional:
            self.right_ptr_rnn = UniBoundaryPointer(mode, input_size, hidden_size)

        self.dropout = torch.nn.Dropout(p=dropout_p)

    def forward(self, Hr, Hr_mask, h_0=None):
        Hr = self.dropout.forward(Hr)

        # split h_0 to left and right
        h_0_left = h_0
        h_0_right = h_0
        if h_0 is not None and self.bidirectional:
            assert self.hidden_size * 2 == h_0.shape[1]
            h_0_left, h_0_right = list(torch.split(h_0, self.hidden_size, dim=1))

        left_beta = self.left_ptr_rnn.forward(Hr, Hr_mask, h_0_left)
        rtn_beta = left_beta
        if self.bidirectional:
            Hr_inv = masked_flip(Hr, Hr_mask)  # mask don't need to flip
            right_beta_inv = self.right_ptr_rnn.forward(Hr_inv, Hr_mask, h_0_right)
            right_beta = masked_flip(right_beta_inv, Hr_mask, flip_dim=2)

            rtn_beta = (left_beta + right_beta) / 2

        # todo: unexplainable
        new_mask = torch.neg((Hr_mask - 1) * 1e-6)  # mask replace zeros with 1e-6, make sure no gradient explosion
        rtn_beta = rtn_beta + new_mask.unsqueeze(0)

        return rtn_beta


class MyRNNBase(torch.nn.Module):
    """
    RNN with packed sequence and dropout
    Args:
        input_size: The number of expected features in the input x
        hidden_size: The number of features in the hidden state h
        bidirectional: If ``True``, becomes a bidirectional RNN. Default: ``False``
        dropout_p: dropout probability to input data, and also dropout along hidden layers

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output, last_state
        - **output** (seq_len, batch, hidden_size * num_directions): tensor
          containing the output features `(h_t)` from the last layer of the RNN,
          for each t.
        - **last_state** (batch, hidden_size * num_directions): the final hidden state of rnn

    """

    def __init__(self, mode, input_size, hidden_size, num_layers, bidirectional, dropout_p):
        super(MyRNNBase, self).__init__()
        self.mode = mode

        if mode == 'LSTM':
            self.hidden = torch.nn.LSTM(input_size=input_size,
                                        hidden_size=hidden_size,
                                        num_layers=num_layers,
                                        dropout=dropout_p,
                                        bidirectional=bidirectional)
        elif mode == 'GRU':
            self.hidden = torch.nn.GRU(input_size=input_size,
                                       hidden_size=hidden_size,
                                       num_layers=num_layers,
                                       dropout=dropout_p,
                                       bidirectional=bidirectional)
        else:
            raise ValueError('Wrong mode select %s, change to LSTM or GRU' % mode)
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Here we reproduce Keras default initialization weights to initialize Embeddings/LSTM weights
        """
        ih = (param.data for name, param in self.named_parameters() if 'weight_ih' in name)
        hh = (param.data for name, param in self.named_parameters() if 'weight_hh' in name)
        b = (param.data for name, param in self.named_parameters() if 'bias' in name)
        for t in ih:
            torch.nn.init.xavier_uniform(t)
        for t in hh:
            torch.nn.init.orthogonal(t)
        for t in b:
            torch.nn.init.constant(t, 0)

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

        v_sort = v.index_select(1, idx_sort)

        v_pack = torch.nn.utils.rnn.pack_padded_sequence(v_sort, lengths_sort)
        v_dropout = self.dropout.forward(v_pack.data)
        v_pack_dropout = torch.nn.utils.rnn.PackedSequence(v_dropout, v_pack.batch_sizes)

        o_pack_dropout, _ = self.hidden.forward(v_pack_dropout)
        o, _ = torch.nn.utils.rnn.pad_packed_sequence(o_pack_dropout)

        # unsorted o
        o_unsort = o.index_select(1, idx_unsort)  # notice here first dim is seq_len

        # get the last time state
        len_idx = (lengths - 1).view(-1, 1).expand(-1, o_unsort.size(2)).unsqueeze(0)
        len_idx = torch.autograd.Variable(len_idx)
        if v.is_cuda:
            len_idx = len_idx.cuda()
        o_last = o_unsort.gather(0, len_idx)

        # new_mask = generate_mask(lengths_sort, enable_cuda=v.is_cuda)
        # new_mask_unsort = new_mask.index_select(0, idx_unsort)

        return o_unsort, o_last


class AttentionPooling(torch.nn.Module):
    """
    Attnetion-Pooling for pointer net init hidden state generate
    Args:
        input_size: The number of expected features in the input x

    Inputs: input, mask
        - **input** (seq_len, batch, input_size): tensor containing the features
          of the input sequence.
        - **mask** (batch, seq_len): tensor show whether a padding index for each element in the batch.

    Outputs: output, output_mask
        - **output** (batch, input_size): tensor containing the output features
    """

    def __init__(self, input_size):
        super(AttentionPooling, self).__init__()

        self.linear_u = torch.nn.Linear(input_size, input_size)
        self.linear_v = torch.nn.Linear(input_size, input_size)
        self.linear_t = torch.nn.Linear(input_size, 1)

        # todo: replace vr and linear_v with one parameter
        self.vr = torch.nn.Parameter(torch.FloatTensor(1, 1, input_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.vr.size(1))
        self.vr.data.uniform_(-stdv, stdv)

    def forward(self, uq, mask):
        wuq_uq = self.linear_u(uq)
        wvq_vq = self.linear_v(self.vr)

        q_tanh = F.tanh(wuq_uq + wvq_vq)
        q_s = self.linear_t(q_tanh) \
            .squeeze(2) \
            .transpose(0, 1)  # (batch, seq_len)

        alpha = masked_softmax(q_s, mask, dim=1)  # (batch, seq_len)
        rq = torch.bmm(alpha.unsqueeze(1), uq.transpose(0, 1)) \
            .squeeze(1)  # (batch, input_size)
        return rq
