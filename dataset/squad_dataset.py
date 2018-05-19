#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import h5py
import math
import torch
import torch.utils.data
from torch.utils.data.sampler import Sampler, SequentialSampler
import logging
import pandas as pd
from dataset.preprocess_data import PreprocessData
from utils.functions import *

logger = logging.getLogger(__name__)


class SquadDataset:
    """
    dataset module for SQuAD
    """

    def __init__(self, global_config):
        self._data = {}
        self._attr = {}
        self.meta_data = {}
        self.global_config = global_config

        # whether preprocessing squad dataset
        is_exist_dataset_h5 = os.path.exists(self.global_config['data']['dataset_h5'])
        assert is_exist_dataset_h5, 'not found dataset hdf5 file in %s' % self.global_config['data']['dataset_h5']
        self._load_hdf5()

    def _load_hdf5(self):
        """
        load squad hdf5 file
        :return:
        """
        squad_h5_path = self.global_config['data']['dataset_h5']
        with h5py.File(squad_h5_path, 'r') as f:
            f_data = f['data']

            for name in ['train', 'dev']:
                self._data[name] = {}
                for sub_name in ['answer_range', 'samples_id']:
                    self._data[name][sub_name] = np.array(f_data[name][sub_name])

                for sub_name in ['context', 'question']:
                    cur_data = f_data[name][sub_name]
                    self._data[name][sub_name] = {}

                    # 'token', 'pos', 'ent', 'em', 'em_lemma', 'right_space'
                    for subsub_name in cur_data.keys():
                        self._data[name][sub_name][subsub_name] = np.array(cur_data[subsub_name])

            for key, value in f.attrs.items():
                self._attr[key] = value

            # 'id2word', 'id2char', 'id2pos', 'id2ent'
            for key in f['meta_data'].keys():
                self.meta_data[key] = np.array(f['meta_data'][key])
        self._char2id = dict(zip(self.meta_data['id2char'],
                                 range(len(self.meta_data['id2char']))))

    def get_dataloader_train(self, batch_size, num_workers):
        """
        a train data dataloader
        :param batch_size:
        :return:
        """
        return self.get_dataloader(batch_size, 'train', num_workers, shuffle=True)

    def get_dataloader_dev(self, batch_size, num_workers):
        """
        a dev data dataloader
        :param batch_size:
        :return:
        """
        return self.get_dataloader(batch_size, 'dev', num_workers, shuffle=False)

    def get_dataloader(self, batch_size, type, num_workers, shuffle):
        """
        get dataloader on train or dev dataset
        :param batch_size:
        :param type: 'train' or 'dev'
        :return:
        """
        data = self._data[type]
        dataset = CQA_Dataset(data['context'],
                              data['question'],
                              data['answer_range'],
                              self.meta_data,
                              self.global_config['preprocess'])
        if shuffle:
            sampler = SortedBatchSampler(dataset.get_lengths(), batch_size)
        else:
            sampler = SequentialSampler(dataset)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=batch_size,
                                                 sampler=sampler,
                                                 collate_fn=self.collect_fun,
                                                 num_workers=num_workers)
        return dataloader

    def collect_fun(self, batch):
        """
        collect function for DataLoader, will generate char idx currently
        :param batch:
        :return:
        """
        context = []
        context_f = []
        question = []
        question_f = []
        answer_range = []

        for ele in batch:
            context.append(ele[0])
            question.append(ele[1])
            context_f.append(ele[2])
            question_f.append(ele[3])
            answer_range.append(ele[4])

        # word idx
        bat_context, max_ct_len = del_zeros_right(torch.stack(context, dim=0))
        bat_question, max_qt_len = del_zeros_right(torch.stack(question, dim=0))
        bat_answer, _ = del_zeros_right(torch.stack(answer_range, dim=0))

        # additional features
        bat_context_f = None
        bat_question_f = None
        if context_f[0] is not None:
            bat_context_f = torch.stack(context_f, dim=0)[:, 0:max_ct_len, :]
            bat_question_f = torch.stack(question_f, dim=0)[:, 0:max_qt_len, :]

        # generate char idx
        bat_context_char = None
        bat_question_char = None
        if self.global_config['preprocess']['use_char']:
            bat_context_char = self._batch_word_to_char(bat_context)
            bat_question_char = self._batch_word_to_char(bat_question)

        return bat_context, bat_question, bat_context_char, bat_question_char, bat_context_f, bat_question_f, bat_answer

    def get_batch_train(self, batch_size):
        """
        a train data batch

        .. warning::
            This method is now deprecated in favor of
            :func:`get_dataloader_train`.
        """
        return self.get_batch_data(batch_size, 'train')

    def get_batch_dev(self, batch_size):
        """
        development data batch

        .. warning::
            This method is now deprecated in favor of
            :func:`get_dataloader_dev`.
        """
        return self.get_batch_data(batch_size, 'dev')

    def get_batch_data(self, batch_size, type):
        """
        same with BatchSampler

        .. warning::
            This method is now deprecated in favor of
            :func:`BatchSampler` and `get_dataloader`.
        """
        data = self._data[type]
        data_size = len(data['context'])
        i = 0
        while i < data_size:
            j = min(i + batch_size, data_size)
            bat = [data['context'][i:j], data['question'][i:j], data['answer_range'][i:j]]
            bat_tensor = [to_long_tensor(x) for x in bat]

            i = j
            yield bat_tensor

    def get_all_samples_id_train(self):
        return self.get_all_samples_id('train')

    def get_all_samples_id_dev(self):
        return self.get_all_samples_id('dev')

    def get_all_samples_id(self, type):
        """
        get samples id of 'train' or 'dev' data
        :param type:
        :return:
        """
        data = self._data[type]
        return data['samples_id']

    def get_all_ct_right_space_train(self):
        return self.get_all_ct_right_space('train')

    def get_all_ct_right_space_dev(self):
        return self.get_all_ct_right_space('dev')

    def get_all_ct_right_space(self, type):
        data = self._data[type]
        return data['context']['right_space']

    def get_train_batch_cnt(self, batch_size):
        """
        get count of train batches
        :param batch_size: single batch size
        :return: count
        """
        data_size = self._attr['train_size']
        cnt_batch = math.ceil(data_size * 1.0 / batch_size)

        return cnt_batch

    def get_dev_batch_cnt(self, batch_size):
        """
        get count of dev batches
        :param batch_size: single batch size
        :return: count
        """
        data_size = self._attr['dev_size']
        cnt_batch = math.ceil(data_size * 1.0 / batch_size)

        return cnt_batch

    def _batch_word_to_char(self, batch_wordid):
        """
        transform batch with sentence of wordid to batch data with sentence of char id
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

    def gen_batch_with_char(self, batch_data, enable_char, device):
        """
        word batch to generate char barch, also move to device, used in train or valid steps

        .. warning::
            This method is now deprecated in favor of collect function in DataLoader
        """
        batch_data = [del_zeros_right(x)[0] for x in batch_data]

        if not enable_char:
            bat_context, bat_question, bat_answer_range = [x.to(device) for x in batch_data]
            bat_context_char = None
            bat_question_char = None

        else:
            bat_context, bat_question, bat_answer_range = batch_data
            bat_context_char = self._batch_word_to_char(bat_context)
            bat_question_char = self._batch_word_to_char(bat_question)

            bat_context, bat_question, bat_context_char, bat_question_char, bat_answer_range = [x.to(device) for x in
                                                                                                [bat_context,
                                                                                                 bat_question,
                                                                                                 bat_context_char,
                                                                                                 bat_question_char,
                                                                                                 bat_answer_range]]

        return bat_context, bat_question, bat_context_char, bat_question_char, bat_answer_range

    def sentence_id2word(self, s_id):
        """
        transform a sentence with word id to a sentence with real word
        :param s_id:
        :return:
        """
        s = map(lambda id: self.meta_data['id2word'][id], s_id)
        return list(s)

    def sentence_word2id(self, s):
        """
        transform a sentence with word to a sentence with word id
        (Note that it's a slow version when using np.where)
        :param s:
        :return:
        """
        s_id = map(lambda word: np.where(self.meta_data['id2word'] == word)[0][0], s)
        return np.array(list(s_id))

    def word_id2char(self, w_id):
        w = map(lambda id: self.meta_data['id2char'][id], w_id)
        return list(w)

    def word_char2id(self, w):
        if w == PreprocessData.padding:  # not actual word
            return np.ones(1, )  # make sure word length>0 and right encoding, here any none-zero value not effect

        w_id = map(lambda ch: self._char2id[ch], w)
        return np.array(list(w_id))

    def sentence_char2id(self, s, max_len=None):
        s_cid = list(map(lambda w: self.word_char2id(w), s))

        if max_len is None:
            word_len = list(map(lambda x: len(x), s_cid))
            max_len = np.max(word_len)
        s_cid_pad = map(lambda x: np.pad(x, (0, max_len - len(x)), 'constant', constant_values=(0, 0)), s_cid)

        return np.stack(list(s_cid_pad), axis=0)

    def gather_context_seq_len(self, type, steps=None):
        """
        gather the context sequence counts with different lengths
        :param type: 'train' or 'dev' data
        :param steps: set to None means default steps
        :return:
        """
        data = self._data[type]
        context = to_long_tensor(data['context']['token'])
        mask = compute_mask(context)
        lengths = mask.eq(1).long().sum(1).squeeze()
        length_pd = pd.DataFrame(data=lengths.numpy(), columns=['length'])

        if steps is None:
            steps = [0, 100, 200, 300, 400, 500, 600, 700, 800]
        assert len(steps) > 0

        # get step length cnt
        real_step = []
        step_length_cnt = []
        for i in range(1, len(steps)):
            lower_bound = steps[i - 1]
            upper_bound = steps[i]
            assert lower_bound < upper_bound  # [lower_bound, upper_bound)
            real_step.append((lower_bound, upper_bound))

            valid = length_pd[(length_pd['length'] < upper_bound) & (length_pd['length'] >= lower_bound)]
            tmp_cnt = valid.shape[0]
            step_length_cnt.append(tmp_cnt)
        rtn_step_length = list(zip(real_step, step_length_cnt))

        # get all length cnt
        length_cnt = length_pd['length'].value_counts().to_frame(name='cnt')
        length_cnt['length'] = length_cnt.index

        return rtn_step_length, length_cnt

    def gather_answer_seq_len(self, type, max_len=None):
        """
        gather the answer sequence counts with different lengths
        :param type: 'train' or 'dev' data
        :param max_len:
        :return:
        """
        data = self._data[type]
        answer_range = data['answer_range']
        lengths = []
        for i in range(answer_range.shape[0]):
            tmp_lens = []
            for j in range(int(answer_range.shape[1] / 2)):
                if answer_range[i, j * 2] != -1:
                    tmp_lens.append(answer_range[i, j * 2 + 1] - answer_range[i, j * 2] + 1)
            lengths.append(min(tmp_lens))

        length_pd = pd.DataFrame(data=lengths, columns=['length'])

        # get all length cnt
        length_cnt = length_pd['length'].value_counts().to_frame(name='cnt')
        length_cnt['length'] = length_cnt.index
        length_cnt = length_cnt.sort_index()

        if max_len is not None:
            sum_len = length_cnt[length_cnt['length'] >= max_len]['cnt'].sum()
            length_cnt = length_cnt[length_cnt['length'] < max_len]
            length_cnt.loc[max_len] = [sum_len, '>=%d' % max_len]

        return length_cnt


class CQA_Dataset(torch.utils.data.Dataset):
    """
    squad like dataset, used for dataloader
    Args:
        - context: (batch, ct_len)
        - question: (batch, qt_len)
        - answer_range: (batch, ans_len)
    """

    def __init__(self, context, question, answer_range, feature_dict, config):
        self.context = context
        self.question = question
        self.answer_range = answer_range
        self.feature_dict = feature_dict
        self.config = config

        self.lengths = self.get_lengths()

    def __getitem__(self, index):
        cur_context = to_long_tensor(self.context['token'][index])
        cur_question = to_long_tensor(self.question['token'][index])
        cur_answer = to_long_tensor(self.answer_range[index])

        cur_context_f, cur_question_f = self.addition_feature(index)
        return cur_context, cur_question, cur_context_f, cur_question_f, cur_answer

    def __len__(self):
        return self.answer_range.shape[0]

    def get_lengths(self):
        ct_mask = self.context['token'].__ne__(PreprocessData.padding_idx)
        ct_lengths = ct_mask.sum(1)

        qt_mask = self.question['token'].__ne__(PreprocessData.padding_idx)
        qt_lengths = qt_mask.sum(1)

        lengths = np.stack([ct_lengths, qt_lengths])

        return lengths

    def addition_feature(self, index):
        data = [self.context, self.question]
        add_features = [None, None]

        for k in range(len(data)):
            features = {}
            tmp_seq_len = data[k]['token'].shape[1]

            if self.config['use_pos']:
                features['pos'] = torch.zeros((tmp_seq_len, len(self.feature_dict['id2pos'])), dtype=torch.float)
                for i, ele in enumerate(data[k]['pos'][index]):
                    if ele == PreprocessData.padding_idx:
                        break
                    features['pos'][i, ele] = 1

            if self.config['use_ent']:
                features['ent'] = torch.zeros((tmp_seq_len, len(self.feature_dict['id2ent'])), dtype=torch.float)
                for i, ele in enumerate(data[k]['ent'][index]):
                    if ele == PreprocessData.padding_idx:
                        break
                    features['ent'][i, ele] = 1

            if self.config['use_em']:
                features['em'] = to_float_tensor(data[k]['em'][index]).unsqueeze(-1)
            if self.config['use_em_lemma']:
                features['em_lemma'] = to_float_tensor(data[k]['em_lemma'][index]).unsqueeze(-1)

            if len(features) > 0:
                add_features[k] = torch.cat(list(features.values()), dim=-1)

        return add_features


class SortedBatchSampler(Sampler):
    """
    forked from https://github.com/HKUST-KnowComp/MnemonicReader
    """

    def __init__(self, lengths, batch_size, shuffle=True):
        self.lengths = lengths  # (2, data_num)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        lengths = np.array(
            [(-l[0], -l[1], np.random.random()) for l in self.lengths.T],
            dtype=[('l1', np.int_), ('l2', np.int_), ('rand', np.float_)]
        )
        indices = np.argsort(lengths, order=('l1', 'l2', 'rand'))
        batches = [indices[i:i + self.batch_size]
                   for i in range(0, len(indices), self.batch_size)]

        last = batches[-1]  # last batch may not be full batch size
        if self.shuffle:
            batches = batches[:len(batches)-1]
            np.random.shuffle(batches)
            batches.append(last)
        return iter([i for batch in batches for i in batch])

    def __len__(self):
        return self.lengths.shape[1]
