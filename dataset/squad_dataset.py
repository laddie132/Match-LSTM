#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import h5py
import numpy as np


class SquadDataset:
    """
    dataset module for SQuAD
    """
    def __init__(self, squad_h5_path='data/squad.h5'):
        self.__data = {}
        self.__attr = {}
        self.load_hdf5(squad_h5_path)

    def load_hdf5(self, squad_h5_path):
        with h5py.File(squad_h5_path, 'r') as f:
            for name in ['train', 'dev']:
                self.__data[name] = {}
                for sub_name in ['context', 'question', 'answer_range']:
                    self.__data[name][sub_name] = np.array(f[name][sub_name])

            for key, value in f.attrs.items():
                self.__attr[key] = value

    def get_batch_gen(self, batch_size):
        """
        a data batch generator
        :param batch_size:
        :return:
        """
        pass

    def get_data(self):
        return self.__data['train'], self.__data['dev']