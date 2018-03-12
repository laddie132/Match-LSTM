#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import h5py
import numpy as np
import torch
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
        self.global_config = global_config

        self.preprocess()       # whether preprocessing squad dataset
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
            batch['context'] = convert_long_variable(train_data['context'][i:j], enable_cuda)
            batch['question'] = convert_long_variable(train_data['question'][i:j], enable_cuda)
            batch['answer_range'] = convert_long_variable(train_data['answer_range'][i:j], enable_cuda)

            batch_data.append(batch)
            i = j

        return batch_data

    def get_dev_data(self, enable_cuda=False):
        """
        development data set
        :param enable_cuda:
        :return:
        """
        dev_data = self.__data['dev']

        dev_data_var = {
            'context': convert_long_variable(dev_data['context'], enable_cuda),
            'question': convert_long_variable(dev_data['question'], enable_cuda),
            'answer_range': convert_long_variable(dev_data['answer_range'], enable_cuda)
        }

        return dev_data_var

    def get_batch_dev(self, batch_size, enable_cuda=False):
        """
        development data batch
        :param enable_cuda:
        :param batch_size:
        :return: [packed squences]
        """
        batch_data = []

        dev_data = self.__data['dev']
        data_size = len(dev_data['context'])
        i = 0
        while i < data_size:
            batch = {}
            j = min(i + batch_size, data_size)
            batch['context'] = convert_long_variable(dev_data['context'][i:j], enable_cuda)
            batch['question'] = convert_long_variable(dev_data['question'][i:j], enable_cuda)
            batch['answer_range'] = convert_long_variable(dev_data['answer_range'][i:j], enable_cuda)

            batch_data.append(batch)
            i = j

        return batch_data

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
