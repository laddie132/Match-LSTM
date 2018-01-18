#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import h5py
import torch
import numpy as np


class GloveEmbedding(torch.nn.Module):
    """
    input: w
    output: x1, x2, ..., xn
    """

    def __init__(self, glove_h5_path):
        super(GloveEmbedding, self).__init__()
        self.glove_h5_path = glove_h5_path
        self.n_embeddings, self.len_embedding, self.weights = self.load_glove_hdf5()

        self.embedding_layer = torch.nn.Embedding(num_embeddings=self.n_embeddings, embedding_dim=self.len_embedding)
        self.embedding_layer.weight = torch.nn.Parameter(self.weights)
        self.embedding_layer.weight.requires_grad = False

    def load_glove_hdf5(self):
        with h5py.File(self.glove_h5_path, 'r') as f:
            id2vec = np.array(f['id2vec'])
            word_dict_size = f.attrs['word_dict_size']
            embedding_size = f.attrs['embedding_size']

        return int(embedding_size), int(word_dict_size), torch.from_numpy(id2vec)

    def forward(self, x):
        return self.embedding_layer.forward(x)  # todo: 去掉padding加的冗余后缀
