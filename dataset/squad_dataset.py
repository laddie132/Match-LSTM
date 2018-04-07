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
from utils.functions import *

logger = logging.getLogger(__name__)


class SquadDataset:
    """
    dataset module for SQuAD
    """

    def __init__(self, global_config):
        self.__data = {}
        self.__attr = {}
        self.__meta_data = {}

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

            self.__meta_data['id2word'] = np.array(f['meta_data']['id2word'])
            self.__meta_data['id2char'] = np.array(f['meta_data']['id2char'])
        self.__meta_data['char2id'] = dict(zip(self.__meta_data['id2char'],
                                               range(len(self.__meta_data['id2char']))))

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

    def get_batch_train(self, batch_size):
        """
        a train data batch
        :param batch_size:
        :return:
        """
        return self.get_batch_data(batch_size, 'train')

    def get_batch_dev(self, batch_size):
        """
        development data batch
        :param batch_size:
        :return: iterator
        """
        return self.get_batch_data(batch_size, 'dev')

    def get_batch_data(self, batch_size, type):
        """
        get batch data
        :param batch_size:
        :return: iterator
        """
        data = self.__data[type]
        data_size = len(data['context'])
        i = 0
        while i < data_size:
            j = min(i + batch_size, data_size)
            bat = [data['context'][i:j], data['question'][i:j], data['answer_range'][i:j]]
            bat_tensor = [to_long_tensor(x) for x in bat]
            bat_tensor_new = [del_zeros_right(x) for x in bat_tensor]

            # bat_context_char = self.batch_word_to_char(bat_tensor_new[0])
            # bat_question_char = self.batch_word_to_char(bat_tensor_new[1])

            # bat_tensor_new.append(bat_context_char)
            # bat_tensor_new.append(bat_question_char)

            i = j
            yield bat_tensor_new

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

    def batch_word_to_char(self, batch_wordid):
        """
        transform batch data with sentence of wordid to batch data wih sentence of char id
        :param batch_wordid: (batch, seq_len), torch tensor
        :return: (batch, seq_len, word_len), torch tensor
        """
        batch_wordid = batch_wordid.numpy()
        batch_word = [self.sentence_id2word(x) for x in batch_wordid]

        batch_length = [[len(x) if x != PreprocessData.padding else 0 for x in s] for s in batch_word]
        batch_max_len = np.max(batch_length)

        batch_char = list(map(lambda x: self.sentence_char2id(x, max_len=batch_max_len), batch_word))
        batch_char = np.stack(batch_char, axis=0)

        return to_long_tensor(batch_char)

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
        s = map(lambda id: self.__meta_data['id2word'][id], s_id)
        return list(s)

    def sentence_word2id(self, s):
        """
        transform a sentence with word to a sentence with word id
        (Notice it's a slow version when using np.where)
        :param s:
        :return:
        """
        s_id = map(lambda word: np.where(self.__meta_data['id2word'] == word)[0][0], s)
        return np.array(list(s_id))

    def word_id2char(self, w_id):
        w = map(lambda id: self.__meta_data['id2char'][id], w_id)
        return list(w)

    def word_char2id(self, w):
        if w == PreprocessData.padding:     # not actual word
            return np.ones(1,)  # make sure word length>0 and right encoding, here any none-zero value not effect

        w_id = map(lambda ch: self.__meta_data['char2id'][ch], w)
        return np.array(list(w_id))

    def sentence_char2id(self, s, max_len=None):
        s_cid = list(map(lambda w: self.word_char2id(w), s))

        if max_len is None:
            word_len = list(map(lambda x: len(x), s_cid))
            max_len = np.max(word_len)
        s_cid_pad = map(lambda x: np.pad(x, (0, max_len-len(x)), 'constant', constant_values=(0, 0)), s_cid)

        return np.stack(list(s_cid_pad), axis=0)


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
