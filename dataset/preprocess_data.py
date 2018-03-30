#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import zipfile
import nltk
import json
import h5py
import logging
import numpy as np
from functools import reduce
from utils.utils import pad_sequences

logger = logging.getLogger(__name__)


class PreprocessData:
    """
    preprocess dataset and glove embedding to hdf5 files
    """

    padding = '__padding__'  # id = 0
    padding_idx = 0

    __compress_option = dict(compression="gzip", compression_opts=9, shuffle=False)

    def __init__(self, global_config):
        # data config
        self.__dev_path = ''
        self.__train_path = ''
        self.__export_squad_path = ''
        self.__glove_path = ''
        self.__embedding_size = 300
        self.__load_config(global_config)

        # preprocess config
        self.__max_context_token_len = 0
        self.__max_question_token_len = 0
        self.__max_answer_len = 0

        # temp data
        self.__word2id = {self.padding: 0}
        self.__word2vec = {self.padding: [0. for i in range(self.__embedding_size)]}
        self.__oov_num = 0

        # data need to store in hdf5 file
        self.__meta_data = {'id2vec': [[0. for i in range(self.__embedding_size)]],
                            'id2word': [self.padding]}
        self.__data = {}
        self.__attr = {}

    def __load_config(self, global_config):
        """
        load config from a dictionary, such as dataset path
        :param global_config: dictionary
        :return:
        """
        data_config = global_config['data']
        self.__train_path = data_config['dataset']['train_path']
        self.__dev_path = data_config['dataset']['dev_path']
        self.__export_squad_path = data_config['dataset_h5']
        self.__glove_path = data_config['embedding_path']
        self.__embedding_size = int(global_config['model']['embedding_size'])

    def __read_json(self, path):
        """
        read json format file from raw squad text
        :param path: squad file path
        :return:
        """
        with open(path, 'r') as f:
            data = json.load(f)

        version = data['version']
        data_list_tmp = [ele['paragraphs'] for ele in data['data']]
        contexts_qas = reduce(lambda a, b: a + b, data_list_tmp)

        self.__attr['dataset_name'] = 'squad-' + version
        return contexts_qas

    def __build_data(self, contexts_qas):
        """
        handle squad data to (context, question, answer_range) with word id representation
        :param contexts_qas: a context with several question-answers
        :return:
        """
        contexts_wid = []
        questions_wid = []
        answers_range_wid = []  # each answer use the [start,end] representation, all the answer horizontal concat

        for question_grp in contexts_qas:
            cur_context = question_grp['context']
            cur_qas = question_grp['qas']

            cur_context_toke = nltk.word_tokenize(cur_context)
            cur_context_ids = self.__sentence_to_id(cur_context_toke)
            self.__max_context_token_len = max(self.__max_context_token_len, len(cur_context_ids))

            for qa in cur_qas:
                cur_question = qa['question']
                cur_question_toke = nltk.word_tokenize(cur_question)
                cur_question_ids = self.__sentence_to_id(cur_question_toke)
                self.__max_question_token_len = max(self.__max_question_token_len, len(cur_question_ids))

                contexts_wid.append(cur_context_ids)
                questions_wid.append(cur_question_ids)

                # find all the answer positions
                cur_answers = qa['answers']
                self.__max_answer_len = max(self.__max_answer_len, len(cur_answers) * 2)

                cur_ans_range_ids = [0 for i in range(len(cur_answers) * 2)]
                for idx, cur_ans in enumerate(cur_answers):
                    cur_ans_start = cur_ans['answer_start']
                    cur_ans_text = cur_ans['text']

                    # add '.' help word tokenize
                    prev_context_token = nltk.word_tokenize(cur_context[:cur_ans_start] + '.')
                    if prev_context_token[len(prev_context_token)-1] == '.':
                        prev_context_token = prev_context_token[:len(prev_context_token) - 1]

                    # if answer start word not the same in context
                    last_pos = len(prev_context_token)-1
                    while len(prev_context_token) > 0 and prev_context_token[last_pos] != cur_context_toke[last_pos]:
                        prev_context_token = prev_context_token[:last_pos]
                        last_pos = len(prev_context_token) - 1
                    pos_s = len(prev_context_token)

                    # find answer end position
                    pos_e = 0
                    tmp_str = ""
                    tmp_ans = cur_ans_text.replace(' ', '').replace('\u202f', '')
                    if tmp_ans[0] == '.':           # squad dataset have some mistakes
                        tmp_ans = tmp_ans[1:]

                    for i in range(pos_s, len(cur_context_toke)):
                        s = cur_context_toke[i]
                        if s == '``' or s == "''":      # because nltk word_tokenize will replace them
                            s = '"'

                        tmp_str += s
                        if tmp_ans in tmp_str:
                            pos_e = i
                            break

                    if pos_e < pos_s:
                        logger.error("Answer start position can't bigger than end position." +
                                     "\nContext:" + cur_context +
                                     "\nQuestion:" + cur_question +
                                     "\nAnswer:" + cur_ans_text)

                    cur_ans_range_ids[(idx * 2):(idx * 2 + 2)] = [pos_s, pos_e]

                answers_range_wid.append(cur_ans_range_ids)

        return {'context': contexts_wid,
                'question': questions_wid,
                'answer_range': answers_range_wid}

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

    def __handle_glove(self):
        """
        handle glove embeddings, restore embeddings with dictionary
        :return:
        """
        logger.debug("read glove from text file %s" % self.__glove_path)
        with zipfile.ZipFile(self.__glove_path, 'r') as zf:
            if len(zf.namelist()) != 1:
                raise ValueError('glove file "%s" not recognized' % self.__glove_path)

            glove_name = zf.namelist()[0]

            word_num = 0
            with zf.open(glove_name) as f:
                for line in f:
                    line_split = line.decode('utf-8').split(' ')
                    self.__word2vec[line_split[0]] = [float(x) for x in line_split[1:]]

                    word_num += 1
                    if word_num % 10000 == 0:
                        logger.debug('handle word No.%d' % word_num)

    def __export_squad_hdf5(self):
        """
        export squad dataset to hdf5 file
        :return:
        """
        f = h5py.File(self.__export_squad_path, 'w')
        str_dt = h5py.special_dtype(vlen=str)

        # attributes
        for attr_name in self.__attr:
            f.attrs[attr_name] = self.__attr[attr_name]

        # meta_data
        id2word = np.array(self.__meta_data['id2word'], dtype=np.str)
        id2vec = np.array(self.__meta_data['id2vec'], dtype=np.float32)
        f_meta_data = f.create_group('meta_data')

        meta_data = f_meta_data.create_dataset('id2word', id2word.shape, dtype=str_dt, **self.__compress_option)
        meta_data[...] = id2word

        meta_data = f_meta_data.create_dataset('id2vec', id2vec.shape, dtype=id2vec.dtype, **self.__compress_option)
        meta_data[...] = id2vec

        # data
        f_data = f.create_group('data')
        for key, value in self.__data.items():
            data_grp = f_data.create_group(key)

            for sub_key, sub_value in value.items():
                data = data_grp.create_dataset(sub_key, sub_value.shape, dtype=sub_value.dtype,
                                               **self.__compress_option)
                data[...] = sub_value

        f.flush()
        f.close()

    def run(self):
        """
        main function to generate hdf5 file
        :return:
        """
        logger.info('handle glove file...')
        self.__handle_glove()

        logger.info('read squad json...')
        train_context_qas = self.__read_json(self.__train_path)
        dev_context_qas = self.__read_json(self.__dev_path)

        logger.info('transform word to id...')
        train_cache_nopad = self.__build_data(train_context_qas)
        dev_cache_nopad = self.__build_data(dev_context_qas)

        self.__attr['train_size'] = len(train_cache_nopad['answer_range'])
        self.__attr['dev_size'] = len(dev_cache_nopad['answer_range'])
        self.__attr['word_dict_size'] = len(self.__word2id)
        self.__attr['embedding_size'] = self.__embedding_size
        self.__attr['oov_word_num'] = self.__oov_num

        logger.info('padding id vectors...')
        self.__data['train'] = {
            'context': pad_sequences(train_cache_nopad['context'], maxlen=self.__max_context_token_len,
                                     padding='post'),
            'question': pad_sequences(train_cache_nopad['question'], maxlen=self.__max_question_token_len,
                                      padding='post'),
            'answer_range': pad_sequences(train_cache_nopad['answer_range'], maxlen=2, padding='post')}
        self.__data['dev'] = {
            'context': pad_sequences(dev_cache_nopad['context'], maxlen=self.__max_context_token_len,
                                     padding='post'),
            'question': pad_sequences(dev_cache_nopad['question'], maxlen=self.__max_question_token_len,
                                      padding='post'),
            'answer_range': pad_sequences(dev_cache_nopad['answer_range'], maxlen=self.__max_answer_len,
                                          padding='post')}

        logger.info('export to hdf5 file...')
        self.__export_squad_hdf5()

        logger.info('finished.')
