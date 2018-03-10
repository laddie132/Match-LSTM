#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import h5py
import numpy as np
import torch


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
            f_data = f['data']

            for name in ['train', 'dev']:
                self.__data[name] = {}
                for sub_name in ['context', 'question', 'answer_range']:
                    self.__data[name][sub_name] = np.array(f_data[name][sub_name])

            for key, value in f.attrs.items():
                self.__attr[key] = value

    def get_batch_train(self, batch_size, enable_cuda=False):
        """
        a train data batch
        :param enable_cuda:
        :param batch_size:
        :return:
        """
        batch_data = []

        train_data = self.__data['train']
        data_size = len(train_data['context'])
        i = 0
        while i < data_size:
            batch = {}
            j = min(i + batch_size, data_size)
            batch['context'] = self.__convert_variable(train_data['context'][i:j], enable_cuda)
            batch['question'] = self.__convert_variable(train_data['question'][i:j], enable_cuda)
            batch['answer_range'] = self.__convert_variable(train_data['answer_range'][i:j], enable_cuda)

            batch_data.append(batch)
            i = j

        return batch_data

    def __convert_variable(self, np_array, enable_cuda=False):
        if enable_cuda:
            return torch.autograd.Variable(torch.from_numpy(np_array).type(torch.LongTensor)).cuda()

        return torch.autograd.Variable(torch.from_numpy(np_array).type(torch.LongTensor))

    def get_dev_data(self, enable_cuda=False):
        """
        development data set
        :param enable_cuda:
        :return:
        """
        dev_data = self.__data['dev']

        dev_data_var = {
            'context': self.__convert_variable(dev_data['context'], enable_cuda),
            'question': self.__convert_variable(dev_data['question'], enable_cuda),
            'answer_range': self.__convert_variable(dev_data['answer_range'], enable_cuda)
        }

        return dev_data_var

    def get_batch_dev(self, batch_size, enable_cuda=False):
        """
        development data batch
        :param enable_cuda:
        :param batch_size:
        :return:
        """
        batch_data = []

        dev_data = self.__data['dev']
        data_size = len(dev_data['context'])
        i = 0
        while i < data_size:
            batch = {}
            j = min(i + batch_size, data_size)
            batch['context'] = self.__convert_variable(dev_data['context'][i:j], enable_cuda)
            batch['question'] = self.__convert_variable(dev_data['question'][i:j], enable_cuda)
            batch['answer_range'] = self.__convert_variable(dev_data['answer_range'][i:j], enable_cuda)

            batch_data.append(batch)
            i = j

        return batch_data