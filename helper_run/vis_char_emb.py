#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import argparse
import torch
import h5py
import string
import numpy as np
import pandas as pd
import hypertools as hyp

sys.path.append(os.getcwd())


def read_label(squad_h5_path):
    with h5py.File(squad_h5_path, 'r') as f:
        id2char = np.array(f['meta_data']['id2char'])
    return id2char


def read_char_emb(weight_path):
    weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
    return weight['char_embedding.embedding_layer.weight'].numpy()


def main(squad_h5_path, weight_path):
    char_label = read_label(squad_h5_path)
    char_emb = read_char_emb(weight_path)

    show_char = list(string.ascii_letters + string.digits)
    show_id = np.where(np.isin(char_label, show_char))[0]

    show_emb = char_emb[show_id]
    show_label = char_label[show_id]

    hyp.plot(show_emb, '.', labels=show_label, n_clusters=10, reduce='TSNE', align='hyper')


def export_csv(squad_h5_path, weight_path):
    """
    used for http://projector.tensorflow.org/
    :param squad_h5_path:
    :param weight_path:
    :return:
    """
    char_label = read_label(squad_h5_path)
    char_emb = read_char_emb(weight_path)

    show_char = list(string.ascii_letters + string.digits)
    show_id = np.where(np.isin(char_label, show_char))[0]

    show_emb = char_emb[show_id]
    show_label = char_label[show_id]

    emb_df = pd.DataFrame(show_emb)
    emb_df.to_csv('data/char_emb.csv', sep='\t', index=False, header=False)

    label_df = pd.DataFrame(show_label)
    label_df.to_csv('data/char_label.csv', sep='\t', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('visual char embedding')
    parser.add_argument('-d', '--data', default='data/squad_glove.h5', dest='squad_h5_path')
    parser.add_argument('-w', '--weight', default='data/model-weight.pt', dest='weight_path')
    args = parser.parse_args()

    # main(args.squad_h5_path, args.weight_path)    # not well intuitive
    export_csv(args.squad_h5_path, args.weight_path)