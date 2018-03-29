#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import h5py
import math
import torch
import torch.utils.data
import logging
from dataset.preprocess_data import PreprocessData
from utils.utils import *

logger = logging.getLogger(__name__)


class SquadDataset:
    """
    dataset module for SQuAD
    """

    def __init__(self, global_config):
        self.__data = {}
        self.__attr = {}
        self.__id2word = []

        self.global_config = global_config

        self.preprocess()  # whether preprocessing squad dataset
        self.load_hdf5()

    def load_hdf5(self):
        """
        load squad hdf5 file
        :return:
        """
        squad_h5_path = self.global_config['data']['dataset_h5']
        with h5py.File(squad_h5_path, 'r') as f:
            f_data = f['data']

            for name in ['train', 'dev']:
                self.__data[name] = {}
                for sub_name in ['context', 'question', 'answer_range']:
                    self.__data[name][sub_name] = np.array(f_data[name][sub_name])

            for key, value in f.attrs.items():
                self.__attr[key] = value

            self.__id2word = np.array(f['meta_data']['id2word'])

    def get_dataloader_train(self, batch_size):
        """
        a train data dataloader
        :param batch_size:
        :return:
        """
        return self.get_dataloader(batch_size, 'train')

    def get_dataloader_dev(self, batch_size):
        """
        a dev data dataloader
        :param batch_size:
        :return:
        """
        return self.get_dataloader(batch_size, 'dev')

    def get_dataloader(self, batch_size, type):
        """
        get dataloader on train or dev dataset
        :param batch_size:
        :param type: 'train' or 'dev'
        :return:
        """
        data = self.__data[type]
        dataset = CQA_Dataset(to_long_tensor(data['context']),
                              to_long_tensor(data['question']),
                              to_long_tensor(data['answer_range']))
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 shuffle=True)
        return dataloader

    def get_batch_train(self, batch_size, enable_cuda=False):
        """
        - notice: replaced by dataloader
        a train data batch
        :param enable_cuda:
        :param batch_size:
        :return:
        """
        train_data = self.__data['train']
        data_size = len(train_data['context'])
        i = 0
        while i < data_size:
            batch = {}
            j = min(i + batch_size, data_size)
            batch['context'] = to_long_variable(train_data['context'][i:j], enable_cuda)
            batch['question'] = to_long_variable(train_data['question'][i:j], enable_cuda)
            batch['answer_range'] = to_long_variable(train_data['answer_range'][i:j], enable_cuda)

            i = j
            yield batch

    def get_batch_dev(self, batch_size, enable_cuda=False):
        """
        - notice: replaced by dataloader
        development data batch
        :param enable_cuda:
        :param batch_size:
        :return: generator [packed squences]
        """
        dev_data = self.__data['dev']
        data_size = len(dev_data['context'])
        i = 0
        while i < data_size:
            batch = {}
            j = min(i + batch_size, data_size)
            batch['context'] = to_long_variable(dev_data['context'][i:j], enable_cuda)
            batch['question'] = to_long_variable(dev_data['question'][i:j], enable_cuda)
            batch['answer_range'] = to_long_variable(dev_data['answer_range'][i:j], enable_cuda)

            i = j
            yield batch

    def get_train_batch_cnt(self, batch_size):
        """
        get count of train batches
        :param batch_size: single batch size
        :return: count
        """
        data_size = self.__attr['train_size']
        cnt_batch = math.ceil(data_size * 1.0 / batch_size)

        return cnt_batch

    def get_dev_batch_cnt(self, batch_size):
        """
        get count of dev batches
        :param batch_size: single batch size
        :return: count
        """
        data_size = self.__attr['dev_size']
        cnt_batch = math.ceil(data_size * 1.0 / batch_size)

        return cnt_batch

    def preprocess(self):
        """
        preprocessing dataset to h5 file
        :return:
        """
        is_exist_dataset_h5 = os.path.exists(self.global_config['data']['dataset_h5'])
        logger.info('%s dataset hdf5 file' % ("found" if is_exist_dataset_h5 else "not found"))

        if not is_exist_dataset_h5:
            logger.info('preprocess data...')
            pdata = PreprocessData(self.global_config)
            pdata.run()

    def sentence_id2word(self, s_id):
        """
        transform a sentence with word id to a sentence with real word
        :param s_id:
        :return:
        """
        s = map(lambda id: self.__id2word[id], s_id)
        return list(s)

    def sentence_word2id(self, s):
        s_id = map(lambda word: np.where(self.__id2word == word)[0][0], s)
        return np.array(list(s_id))


class CQA_Dataset(torch.utils.data.Dataset):
    """
    torch dataset type, used for dataloader
    """
    def __init__(self, context, question, answer_range):
        self.context = context
        self.question = question
        self.answer_range = answer_range

    def __getitem__(self, index):
        return self.context[index], self.question[index], self.answer_range[index]

    def __len__(self):
        return self.answer_range.shape[0]
