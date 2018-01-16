#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import zipfile
import nltk
import json
import h5py
import logging
import numpy as np
from pprint import pprint
from functools import reduce
from utils.load_config import init_logging

init_logging()
logger = logging.getLogger(__name__)


class PreprocessData:
    dev_path = 'data/SQuAD/dev-v1.1.json'
    train_path = 'data/SQuAD/train-v1.1.json'
    export_path = 'data/squad.h5'

    glove_path = 'data/glove.840B.300d.zip'

    padding = '__padding__'     # id = 0
    oov = '__oov___'            # id = 1
    embedding_size = 300

    compress_option = dict(compression="gzip", compression_opts=9, shuffle=False)

    def __init__(self):
        self.word2id = {}
        self.max_context_token_len = 0
        self.max_question_token_len = 0
        self.max_answer_len = 0
        self.caches = {}

        '''need to store in hdf5 file'''
        self.meta_data = {}
        self.data = {}
        self.attributes = {}

    def read_json(self, path):
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

        self.attributes['name'] = 'squad-' + version
        return contexts_qas

    def build_data(self, contexts_qas):
        """
        handle squad data to (context, question, answer_range) with word id representation
        :param contexts_qas: a context with several question-answers
        :return:
        """
        contexts_wid = []
        questions_wid = []
        answers_range_wid = []      # each answer use the [start,end) representation, all the answer horizontal concat

        for question_grp in contexts_qas:
            cur_context = question_grp['context']
            cur_qas = question_grp['qas']

            cur_context_toke = nltk.word_tokenize(cur_context)
            cur_context_ids = self.sentence_to_id(cur_context_toke)
            self.max_context_token_len = max(self.max_context_token_len, len(cur_context_ids))

            for qa in cur_qas:
                cur_question = qa['question']
                cur_question_toke = nltk.word_tokenize(cur_question)
                cur_question_ids = self.sentence_to_id(cur_question_toke)
                self.max_question_token_len = max(self.max_question_token_len, len(cur_question_ids))

                contexts_wid.append(cur_context_ids)
                questions_wid.append(cur_question_ids)

                # find all the answer positions
                cur_answers = qa['answers']
                self.max_answer_size = max(self.max_answer_size, len(cur_answers)*2)

                cur_ans_range_ids = [0 for i in range(len(cur_answers) * 2)]
                for idx, cur_ans in enumerate(cur_answers):
                    cur_ans_text = nltk.word_tokenize(cur_ans['text'])
                    pos_s = self.find_sublist(cur_ans_text, cur_context_toke)   # not consider find multi position in context
                    pos_e = pos_s + len(cur_ans_text)

                    cur_ans_range_ids[(idx*2):(idx*2+2)] = [pos_s, pos_e]

                answers_range_wid.append(cur_ans_range_ids)

        return {'context': contexts_wid,
                'question': questions_wid,
                'answer_range': answers_range_wid}

    def find_sublist(self, query, base):
        """
        find sublist`s start position in a base list
        :param query: query sublist
        :param base: base list
        :return:
        """
        for i in range(len(base)):
            if base[i:(i+len(query))] == query:
                return i

        return -1

    def sentence_to_id(self, sentence):
        """
        transform a sentence to word index id representation
        :param sentence: tokenized sentence
        :return: word ids
        """

        ids = []
        for word in sentence:
            if word in self.word2id:
                ids.append(self.word2id[word])
            else:
                ids.append(self.word2id[self.oov])

        return ids

    def handle_glove(self):
        """
        transform glove embeddings to meta_data in hdf5 file
        :return:
        """
        with zipfile.ZipFile(self.glove_path, 'r') as zf:
            if len(zf.namelist()) != 1:
                raise ValueError('glove file "%s" not recognized' % self.glove_path)

            glove_name = zf.namelist()[0]

            words = [self.padding, self.oov]
            embeddings = [[0. for i in range(self.embedding_size)]]

            word_num = 0
            with zf.open(glove_name) as f:
                for line in f:
                    line_split = line.decode('utf-8').split(' ')
                    words.append(line_split[0])
                    embeddings.append([float(x) for x in line_split[1:]])

                    word_num += 1
                    if word_num % 10000 == 0:
                        logger.debug('handle word No.%d' % word_num)

            self.meta_data['id2word'] = np.array(words, dtype=np.str)
            self.meta_data['id2vec'] = np.array(embeddings, dtype=np.float32)
            self.word2id = dict(zip(words, range(len(words))))

    def export_hdf5(self):
        """
        export the attributes, meta_data and data to hdf5 file
        :return:
        """
        f = h5py.File(self.export_path, 'w')
        str_dt = h5py.special_dtype(vlen=str)

        # attributes
        for attr_name in self.attributes:
            f.attrs[attr_name] = self.attributes[attr_name]

        # data
        data_grp = f.create_group('data')
        for key, value in self.data.items():
            data_grp_sub = data_grp.create_group(key)

            for sub_key, sub_value in value.items():
                data_grp_sub_data = data_grp_sub.create_dataset(sub_key, sub_value.shape, dtype=sub_value.dtype, **self.compress_option)
                data_grp_sub_data[...] = sub_value

        # meta_data
        meta_data_grp = f.create_group('meta_data')
        for key, value in self.meta_data.items():
            dt = value.dtype
            if type(value[0]) == np.str_:
                dt = str_dt
            meta_data_grp_data = meta_data_grp.create_dataset(key, value.shape, dtype=dt, **self.compress_option)
            meta_data_grp_data[...] = value

        f.flush()
        f.close()

    def run(self):
        """
        main function to generate hdf5 file
        :return:
        """
        logger.info('handle glove file...')
        self.handle_glove()

        logger.info('read squad json...')
        train_context_qas = self.read_json(self.train_path)
        dev_context_qas = self.read_json(self.dev_path)

        logger.info('transform word to id...')
        train_cache_nopad = self.build_data(train_context_qas)
        dev_cache_nopad = self.build_data(dev_context_qas)

        self.attributes['train_size'] = len(train_cache_nopad['answer_range'])
        self.attributes['dev_size'] = len(dev_cache_nopad['answer_range'])

        logger.info('padding id vectors...')
        self.data['train'] = {
            'context': self.pad_sequences(train_cache_nopad['context'], maxlen=self.max_context_token_len, padding='post'),
            'question': self.pad_sequences(train_cache_nopad['question'], maxlen=self.max_question_token_len, padding='post'),
            'answer_range': self.pad_sequences(train_cache_nopad['answer_range'], maxlen=1, padding='post')}

        self.data['dev'] = {
            'context': self.pad_sequences(dev_cache_nopad['context'], maxlen=self.max_context_token_len, padding='post'),
            'question': self.pad_sequences(dev_cache_nopad['question'], maxlen=self.max_question_token_len, padding='post'),
            'answer_range': self.pad_sequences(dev_cache_nopad['answer_range'], maxlen=self.max_answer_size, padding='post')}

        logger.info('export to hdf5 file...')
        self.export_hdf5()

        logger.info('finished.')

    def pad_sequences(self, sequences, maxlen=None, dtype='int32', padding='pre', truncating='pre', value=0.):
        '''
        FROM KERAS
        Pads each sequence to the same length:
        the length of the longest sequence.
        If maxlen is provided, any sequence longer
        than maxlen is truncated to maxlen.
        Truncation happens off either the beginning (default) or
        the end of the sequence.
        Supports post-padding and pre-padding (default).
        # Arguments
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger than
                maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.
        # Returns
            x: numpy array with dimensions (number_of_sequences, maxlen)
        '''
        lengths = [len(s) for s in sequences]

        nb_samples = len(sequences)
        if maxlen is None:
            maxlen = np.max(lengths)

        # take the sample shape from the first non empty sequence
        # checking for consistency in the main loop below.
        sample_shape = tuple()
        for s in sequences:
            if len(s) > 0:
                sample_shape = np.asarray(s).shape[1:]
                break

        x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
        for idx, s in enumerate(sequences):
            if len(s) == 0:
                continue  # empty list was found
            if truncating == 'pre':
                trunc = s[-maxlen:]
            elif truncating == 'post':
                trunc = s[:maxlen]
            else:
                raise ValueError('Truncating type "%s" not understood' % truncating)

            # check `trunc` has expected shape
            trunc = np.asarray(trunc, dtype=dtype)
            if trunc.shape[1:] != sample_shape:
                raise ValueError('Shape of sample %s of sequence at position %s is different from expected shape %s' %
                                 (trunc.shape[1:], idx, sample_shape))

            if padding == 'post':
                x[idx, :len(trunc)] = trunc
            elif padding == 'pre':
                x[idx, -len(trunc):] = trunc
            else:
                raise ValueError('Padding type "%s" not understood' % padding)
        return x


if __name__ == '__main__':
    preprocess = PreprocessData()
    preprocess.run()