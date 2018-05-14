#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import zipfile
import jieba
from pyhanlp import *
import json
import h5py
import logging
import numpy as np
from functools import reduce
from utils.functions import pad_sequences

logger = logging.getLogger(__name__)


class PreprocessCMRC:
    """
    preprocess dataset and glove embedding to hdf5 files
    """

    padding = '__padding__'  # id = 0
    padding_idx = 0          # also same to char level padding values
    answer_padding_idx = -1

    __compress_option = dict(compression="gzip", compression_opts=9, shuffle=False)

    def __init__(self, global_config):
        # data config
        self.__dev_path = ''
        self.__train_path = ''
        self.__export_cmrc_path = ''
        self.__embedding_path = ''
        self.__embedding_size = 300
        self.__ignore_max_len = 10000
        self.__load_config(global_config)

        # preprocess config
        self.__max_context_token_len = 0
        self.__max_question_token_len = 0
        self.__max_answer_len = 0

        # temp data
        self.__word2id = {self.padding: 0}
        self.__char2id = {self.padding: 0}
        self.__word2vec = {self.padding: [0. for i in range(self.__embedding_size)]}
        self.__oov_num = 0

        # data need to store in hdf5 file
        self.__meta_data = {'id2vec': [[0. for i in range(self.__embedding_size)]],
                            'id2word': [self.padding],
                            'id2char': [self.padding]}
        self.__data = {}
        self.__attr = {'dataset_name': 'CMRC'}

    def __load_config(self, global_config):
        """
        load config from a dictionary, such as dataset path
        :param global_config: dictionary
        :return:
        """
        data_config = global_config['data']
        self.__train_path = data_config['dataset']['train_path']
        self.__dev_path = data_config['dataset']['dev_path']
        self.__export_cmrc_path = data_config['dataset_h5']
        self.__embedding_path = data_config['embedding_path']
        self.__ignore_max_len = data_config['ignore_max_len']
        self.__embedding_size = int(global_config['model']['encoder']['word_embedding_size'])

    def __build_data(self, path, training):
        """
        handle squad data to (context, question, answer_range) with word id representation
        :param contexts_qas: a context with several question-answers
        :return:
        """
        contexts_wid = []
        questions_wid = []
        answers_range_wid = []  # each answer use the [start,end] representation, all the answer horizontal concat
        samples_id = []

        with open(path, 'r') as f:
            contexts_qas = json.load(f)

        for question_grp in contexts_qas:
            cur_context = question_grp['context_text']
            cur_qas = question_grp['qas']

            cur_context_toke = HanLP.segment(cur_context)
            if training and len(cur_context_toke) > self.__ignore_max_len:  # some context token len too large, such as 766
                continue

            self.__update_to_char(cur_context)
            cur_context_ids = self.__sentence_to_id(cur_context_toke)
            self.__max_context_token_len = max(self.__max_context_token_len, len(cur_context_ids))

            for qa in cur_qas:
                cur_question = qa['query_text']
                self.__update_to_char(cur_question)
                cur_question_toke = HanLP.segment(cur_question)
                cur_question_ids = self.__sentence_to_id(cur_question_toke)
                self.__max_question_token_len = max(self.__max_question_token_len, len(cur_question_ids))

                contexts_wid.append(cur_context_ids)
                questions_wid.append(cur_question_ids)
                samples_id.append(qa['query_id'])

                # find all the answer positions
                cur_answers = qa['answers']
                self.__max_answer_len = max(self.__max_answer_len, len(cur_answers) * 2)

                cur_ans_range_ids = [self.answer_padding_idx for i in range(len(cur_answers) * 2)]
                for idx, cur_ans in enumerate(cur_answers):
                    cur_ans = str(cur_ans)  # it has some numbers
                    cur_ans_len = len(cur_ans)
                    try:
                        char_pos_s = cur_context.index(cur_ans)
                    except ValueError:
                        if sum(cur_ans_range_ids) == self.answer_padding_idx * len(cur_answers) * 2:
                            logger.error('Answer not in context' +
                                         "\nContext:" + cur_context +
                                         "\nQuestion:" + cur_question +
                                         "\nAnswer:" + cur_ans)
                        continue

                    char_pos_e = char_pos_s + cur_ans_len - 1

                    pos_s = 0
                    pos_e = 0
                    is_find_s = False

                    # find start and end position
                    tmp_len = 0
                    for token_i, token in enumerate(cur_context_toke):
                        tmp_len += len(token)

                        if not is_find_s and tmp_len - 1 >= char_pos_s:
                            pos_s = token_i
                            is_find_s = True
                        if tmp_len - 1 >= char_pos_e:
                            pos_e = token_i
                            break

                    if pos_e < pos_s:
                        logger.error("Answer start position can't bigger than end position." +
                                     "\nContext:" + cur_context +
                                     "\nQuestion:" + cur_question +
                                     "\nAnswer:" + cur_ans)

                    gen_ans = ''.join(cur_context_toke[pos_s:(pos_e+1)]).replace(' ', '')
                    true_ans = cur_ans.replace(' ', '')
                    if true_ans not in gen_ans:
                        logger.error("Answer position wrong." +
                                     "\nContext:" + cur_context +
                                     "\nQuestion:" + cur_question +
                                     "\nAnswer:" + cur_ans)

                    cur_ans_range_ids[(idx * 2):(idx * 2 + 2)] = [pos_s, pos_e]

                answers_range_wid.append(cur_ans_range_ids)

        return {'context': contexts_wid,
                'question': questions_wid,
                'answer_range': answers_range_wid,
                'samples_id': samples_id}

    def __sentence_to_id(self, sentence):
        """
        transform a sentence to word index id representation
        :param sentence: tokenized sentence
        :return: word ids
        """

        ids = []
        for word in sentence:
            if word not in self.__word2id:
                self.__word2id[word] = len(self.__word2id)
                self.__meta_data['id2word'].append(word)

                # whether OOV
                if word in self.__word2vec:
                    self.__meta_data['id2vec'].append(self.__word2vec[word])
                else:
                    self.__oov_num += 1
                    logger.debug('No.%d OOV word %s' % (self.__oov_num, word))
                    self.__meta_data['id2vec'].append([0. for i in range(self.__embedding_size)])
            ids.append(self.__word2id[word])

        return ids

    def __update_to_char(self, sentence):
        """
        update char2id
        :param sentence: raw sentence
        """
        for ch in sentence:
            if ch not in self.__char2id:
                self.__char2id[ch] = len(self.__char2id)
                self.__meta_data['id2char'].append(ch)

    def __handle_embeddings(self):
        """
        handle word embeddings, restore embeddings with dictionary
        :return:
        """
        logger.debug("read embeddings from text file %s" % self.__embedding_path)
        if not os.path.exists(self.__embedding_path):
            raise ValueError('embeddings file "%s" not recognized' % self.__embedding_path)

        word_num = 0
        embedding_size = 0
        embedding_num = 0

        with open(self.__embedding_path, encoding='latin-1') as f:
            for line in f:
                line_split = line.strip().split(' ')
                if word_num == 0:
                    embedding_num = int(line_split[0])
                    embedding_size = int(line_split[1])
                    logger.info('Embedding size: %d' % embedding_size)
                    logger.info('Embedding num: %d' % embedding_num)

                else:
                    self.__word2vec[line_split[0]] = [float(x) for x in line_split[1:]]

                word_num += 1
                if word_num % 10000 == 0:
                    logger.info('handle word No.%d' % word_num)


    def __export_cmrc_hdf5(self):
        """
        export squad dataset to hdf5 file
        :return:
        """
        f = h5py.File(self.__export_cmrc_path, 'w')
        str_dt = h5py.special_dtype(vlen=str)

        # attributes
        for attr_name in self.__attr:
            f.attrs[attr_name] = self.__attr[attr_name]

        # meta_data
        id2word = np.array(self.__meta_data['id2word'], dtype=np.str)
        id2char = np.array(self.__meta_data['id2char'], dtype=np.str)
        id2vec = np.array(self.__meta_data['id2vec'], dtype=np.float32)
        f_meta_data = f.create_group('meta_data')

        meta_data = f_meta_data.create_dataset('id2word', id2word.shape, dtype=str_dt, **self.__compress_option)
        meta_data[...] = id2word

        meta_data = f_meta_data.create_dataset('id2char', id2char.shape, dtype=str_dt, **self.__compress_option)
        meta_data[...] = id2char

        meta_data = f_meta_data.create_dataset('id2vec', id2vec.shape, dtype=id2vec.dtype, **self.__compress_option)
        meta_data[...] = id2vec

        # data
        f_data = f.create_group('data')
        for key, value in self.__data.items():
            data_grp = f_data.create_group(key)

            for sub_key, sub_value in value.items():
                cur_dtype = str_dt if sub_value.dtype.type is np.str_ else sub_value.dtype
                data = data_grp.create_dataset(sub_key, sub_value.shape, dtype=cur_dtype,
                                               **self.__compress_option)
                data[...] = sub_value

        f.flush()
        f.close()

    def run(self):
        """
        main function to generate hdf5 file
        :return:
        """
        logger.info('handle embedding file...')
        self.__handle_embeddings()

        logger.info('read squad json...')
        logger.info('transform word to id...')
        train_cache_nopad = self.__build_data(self.__train_path, training=True)
        dev_cache_nopad = self.__build_data(self.__dev_path, training=False)

        self.__attr['train_size'] = len(train_cache_nopad['answer_range'])
        self.__attr['dev_size'] = len(dev_cache_nopad['answer_range'])
        self.__attr['word_dict_size'] = len(self.__word2id)
        self.__attr['char_dict_size'] = len(self.__char2id)
        self.__attr['embedding_size'] = self.__embedding_size
        self.__attr['oov_word_num'] = self.__oov_num

        logger.info('padding id vectors...')
        self.__data['train'] = {
            'context': pad_sequences(train_cache_nopad['context'],
                                     maxlen=self.__max_context_token_len,
                                     padding='post',
                                     value=self.padding_idx),
            'question': pad_sequences(train_cache_nopad['question'],
                                      maxlen=self.__max_question_token_len,
                                      padding='post',
                                      value=self.padding_idx),
            'answer_range': np.array(train_cache_nopad['answer_range']),
            'samples_id': np.array(train_cache_nopad['samples_id'])}
        self.__data['dev'] = {
            'context': pad_sequences(dev_cache_nopad['context'],
                                     maxlen=self.__max_context_token_len,
                                     padding='post',
                                     value=self.padding_idx),
            'question': pad_sequences(dev_cache_nopad['question'],
                                      maxlen=self.__max_question_token_len,
                                      padding='post',
                                      value=self.padding_idx),
            'answer_range': pad_sequences(dev_cache_nopad['answer_range'],
                                          maxlen=self.__max_answer_len,
                                          padding='post',
                                          value=self.answer_padding_idx),
            'samples_id': np.array(dev_cache_nopad['samples_id'])}

        logger.info('export to hdf5 file...')
        self.__export_cmrc_hdf5()

        logger.info('finished.')
