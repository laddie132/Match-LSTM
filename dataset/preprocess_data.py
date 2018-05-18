#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import re
import zipfile
import spacy
import json
import h5py
import logging
import numpy as np
from functools import reduce
from utils.functions import pad_sequences
from .doc_text import DocText, Space

logger = logging.getLogger(__name__)


class PreprocessData:
    """
    preprocess dataset and glove embedding to hdf5 files
    """

    padding = '__padding__'  # id = 0
    padding_idx = 0  # all the features padding idx, exclude answer_range
    answer_padding_idx = -1

    _compress_option = dict(compression="gzip", compression_opts=9, shuffle=False)

    def __init__(self, global_config):
        # data config
        self._dev_path = ''
        self._train_path = ''
        self._export_squad_path = ''
        self._glove_path = ''
        self._embedding_size = 300
        self._ignore_max_len = 10000
        self._load_config(global_config)

        # preprocess config
        self._max_answer_len = 0

        # temp data
        self._word2id = {self.padding: 0}
        self._char2id = {self.padding: 0, '`': 1}  # because nltk word tokenize will replace '"' with '``'
        self._pos2id = {self.padding: 0}
        self._ent2id = {self.padding: 0}
        self._word2vec = {self.padding: [0. for i in range(self._embedding_size)]}
        self._oov_num = 0

        # data need to store in hdf5 file
        self._meta_data = {'id2vec': [[0. for i in range(self._embedding_size)]],
                           'id2word': [self.padding],
                           'id2char': [self.padding, '`'],
                           'id2pos': [self.padding],
                           'id2ent': [self.padding]}
        self._data = {}
        self._attr = {}

        self._nlp = spacy.load('en')
        self._nlp.remove_pipe('parser')
        if not any([self._use_em_lemma, self._use_pos, self._use_ent]):
            self._nlp.remove_pipe('tagger')
        if not self._use_ent:
            self._nlp.remove_pipe('ner')

    def _load_config(self, global_config):
        """
        load config from a dictionary, such as dataset path
        :param global_config: dictionary
        :return:
        """
        data_config = global_config['data']
        self._train_path = data_config['dataset']['train_path']
        self._dev_path = data_config['dataset']['dev_path']
        self._export_squad_path = data_config['dataset_h5']
        self._glove_path = data_config['embedding_path']

        self.preprocess_config = global_config['preprocess']
        self._ignore_max_len = self.preprocess_config['ignore_max_len']
        self._use_char = self.preprocess_config['use_char']
        self._use_pos = self.preprocess_config['use_pos']
        self._use_ent = self.preprocess_config['use_ent']
        self._use_em = self.preprocess_config['use_em']
        self._use_em_lemma = self.preprocess_config['use_em_lemma']

        self._embedding_size = int(self.preprocess_config['word_embedding_size'])

    def _read_json(self, path):
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

        self._attr['dataset_name'] = 'squad-' + version
        return contexts_qas

    def _build_data(self, contexts_qas, training):
        """
        handle squad data to (context, question, answer_range) with word id representation
        :param contexts_qas: a context with several question-answers
        :return:
        """
        contexts_doc = []
        questions_doc = []
        answers_range_wid = []  # each answer use the [start,end] representation, all the answer horizontal concat
        samples_id = []

        cnt = 0

        # every context
        for question_grp in contexts_qas:
            cur_context = question_grp['context']
            cur_qas = question_grp['qas']

            cur_context_doc = DocText(self._nlp, cur_context, self.preprocess_config)
            if training and len(cur_context_doc) > self._ignore_max_len:  # some context token len too large
                continue

            if self._use_char:
                self._update_to_char(cur_context)

            cur_context_ids = self._doctext_to_id(cur_context_doc)

            # every question-answer
            for qa in cur_qas:
                cur_question = qa['question']

                if self._use_char:
                    self._update_to_char(cur_question)

                cur_question_doc = DocText(self._nlp, cur_question, self.preprocess_config)
                cur_question_ids = self._doctext_to_id(cur_question_doc)

                # get em feature
                if self._use_em or self._use_em_lemma:
                    cur_context_doc.update_em(cur_question_doc)
                    cur_question_doc.update_em(cur_context_doc)

                cur_context_ids['em'] = cur_context_doc.em
                cur_context_ids['em_lemma'] = cur_context_doc.em_lemma
                cur_question_ids['em'] = cur_question_doc.em
                cur_question_ids['em_lemma'] = cur_question_doc.em_lemma

                contexts_doc.append(cur_context_ids)
                questions_doc.append(cur_question_ids)
                samples_id.append(qa['id'])

                # find all the answer positions
                cur_answers = qa['answers']
                self._max_answer_len = max(self._max_answer_len, len(cur_answers) * 2)

                cur_ans_range_ids = [0 for i in range(len(cur_answers) * 2)]
                for idx, cur_ans in enumerate(cur_answers):
                    cur_ans_start = cur_ans['answer_start']
                    cur_ans_text = cur_ans['text']

                    pos_s, pos_e = self.find_ans_start_end(cur_context, cur_context_doc, cur_ans_text, cur_ans_start)
                    if pos_e < pos_s:
                        logger.error("Answer start position can't bigger than end position." +
                                     "\nContext:" + cur_context +
                                     "\nQuestion:" + cur_question +
                                     "\nAnswer:" + cur_ans_text)
                        continue

                    gen_ans = ''.join(cur_context_doc.token[pos_s:(pos_e + 1)]).replace(' ', '')
                    true_ans = Space.remove_white_space(cur_ans['text'])
                    if true_ans not in gen_ans:
                        logger.error("Answer position wrong." +
                                     "\nContext:" + cur_context +
                                     "\nQuestion:" + cur_question +
                                     "\nAnswer:" + cur_ans_text)
                        continue

                    cur_ans_range_ids[(idx * 2):(idx * 2 + 2)] = [pos_s, pos_e]
                answers_range_wid.append(cur_ans_range_ids)

                cnt += 1
                if cnt % 100 == 0:
                    logger.info('No.%d sample handled.' % cnt)

        return {'context': contexts_doc,
                'question': questions_doc,
                'answer_range': answers_range_wid,
                'samples_id': samples_id}

    def find_ans_start_end(self, context_text, context_doc, answer_text, answer_start):
        # find answer start position
        pre_ans_len = len(Space.remove_white_space(context_text[:answer_start]))
        tmp_len = 0
        pos_s = 0

        for i in range(len(context_doc)):
            tmp_len += len(context_doc.token[i])

            if tmp_len > pre_ans_len:
                pos_s = i
                break

        # find answer end position
        pos_e = 0
        tmp_str = ""
        tmp_ans = Space.remove_white_space(answer_text)
        if tmp_ans[0] == '.':  # squad dataset have some mistakes
            tmp_ans = tmp_ans[1:]

        for i in range(pos_s, len(context_doc)):
            s = context_doc.token[i]

            tmp_str += s
            if tmp_ans in tmp_str:
                pos_e = i
                break

        return pos_s, pos_e

    def _doctext_to_id(self, doc_text):
        """
        transform a sentence to word index id representation
        :param sentence: DocText
        :return: word ids
        """

        sentence = {'token': [], 'pos': [], 'ent': [], 'right_space': doc_text.right_space}

        for i in range(len(doc_text)):

            # word
            word = doc_text.token[i]
            if word not in self._word2id:
                self._word2id[word] = len(self._word2id)
                self._meta_data['id2word'].append(word)

                # whether OOV
                if word in self._word2vec:
                    self._meta_data['id2vec'].append(self._word2vec[word])
                else:
                    self._oov_num += 1
                    logger.debug('No.%d OOV word %s' % (self._oov_num, word))
                    self._meta_data['id2vec'].append([0. for i in range(self._embedding_size)])
            sentence['token'].append(self._word2id[word])

            # pos
            if self._use_pos:
                pos = doc_text.pos[i]
                if pos not in self._pos2id:
                    self._pos2id[pos] = len(self._pos2id)
                    self._meta_data['id2pos'].append(pos)
                sentence['pos'].append(self._pos2id[pos])

            # ent
            if self._use_ent:
                ent = doc_text.ent[i]
                if ent not in self._ent2id:
                    self._ent2id[ent] = len(self._ent2id)
                    self._meta_data['id2ent'].append(ent)
                sentence['ent'].append(self._ent2id[ent])

        return sentence

    def _update_to_char(self, sentence):
        """
        update char2id
        :param sentence: raw sentence
        """
        for ch in sentence:
            if ch not in self._char2id:
                self._char2id[ch] = len(self._char2id)
                self._meta_data['id2char'].append(ch)

    def _handle_glove(self):
        """
        handle glove embeddings, restore embeddings with dictionary
        :return:
        """
        logger.info("read glove from text file %s" % self._glove_path)
        with zipfile.ZipFile(self._glove_path, 'r') as zf:
            if len(zf.namelist()) != 1:
                raise ValueError('glove file "%s" not recognized' % self._glove_path)

            glove_name = zf.namelist()[0]

            word_num = 0
            with zf.open(glove_name) as f:
                for line in f:
                    line_split = line.decode('utf-8').split(' ')
                    self._word2vec[line_split[0]] = [float(x) for x in line_split[1:]]

                    word_num += 1
                    if word_num % 10000 == 0:
                        logger.info('handle word No.%d' % word_num)

    def _export_squad_hdf5(self):
        """
        export squad dataset to hdf5 file
        :return:
        """
        f = h5py.File(self._export_squad_path, 'w')
        str_dt = h5py.special_dtype(vlen=str)

        # attributes
        for attr_name in self._attr:
            f.attrs[attr_name] = self._attr[attr_name]

        # meta_data
        f_meta_data = f.create_group('meta_data')
        for key in ['id2word', 'id2char', 'id2pos', 'id2ent']:
            value = np.array(self._meta_data[key], dtype=np.str)
            meta_data = f_meta_data.create_dataset(key, value.shape, dtype=str_dt, **self._compress_option)
            meta_data[...] = value

        id2vec = np.array(self._meta_data['id2vec'], dtype=np.float32)
        meta_data = f_meta_data.create_dataset('id2vec', id2vec.shape, dtype=id2vec.dtype, **self._compress_option)
        meta_data[...] = id2vec

        # data
        f_data = f.create_group('data')
        for key, value in self._data.items():
            data_grp = f_data.create_group(key)

            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    sub_grp = data_grp.create_group(sub_key)
                    for subsub_key, subsub_value in sub_value.items():
                        if len(subsub_value) == 0:
                            continue

                        cur_dtype = str_dt if subsub_value.dtype.type is np.str_ else subsub_value.dtype
                        data = sub_grp.create_dataset(subsub_key, subsub_value.shape, dtype=cur_dtype,
                                                      **self._compress_option)
                        data[...] = subsub_value
                else:
                    cur_dtype = str_dt if sub_value.dtype.type is np.str_ else sub_value.dtype
                    data = data_grp.create_dataset(sub_key, sub_value.shape, dtype=cur_dtype,
                                                   **self._compress_option)
                    data[...] = sub_value

        f.flush()
        f.close()

    def run(self):
        """
        main function to generate hdf5 file
        :return:
        """
        logger.info('handle glove file...')
        self._handle_glove()

        logger.info('read squad json...')
        train_context_qas = self._read_json(self._train_path)
        dev_context_qas = self._read_json(self._dev_path)

        logger.info('transform word to id...')
        train_cache_nopad = self._build_data(train_context_qas, training=True)
        dev_cache_nopad = self._build_data(dev_context_qas, training=False)

        self._attr['train_size'] = len(train_cache_nopad['answer_range'])
        self._attr['dev_size'] = len(dev_cache_nopad['answer_range'])
        self._attr['word_dict_size'] = len(self._word2id)
        self._attr['char_dict_size'] = len(self._char2id)
        self._attr['pos_dict_size'] = len(self._pos2id)
        self._attr['ent_dict_size'] = len(self._ent2id)
        self._attr['embedding_size'] = self._embedding_size
        self._attr['oov_word_num'] = self._oov_num

        logger.info('padding id vectors...')
        self._data['train'] = {
            'context': dict2array(train_cache_nopad['context']),
            'question': dict2array(train_cache_nopad['question']),
            'answer_range': np.array(train_cache_nopad['answer_range']),
            'samples_id': np.array(train_cache_nopad['samples_id'])
        }
        self._data['dev'] = {
            'context': dict2array(dev_cache_nopad['context']),
            'question': dict2array(dev_cache_nopad['question']),
            'answer_range': pad_sequences(dev_cache_nopad['answer_range'],
                                          maxlen=self._max_answer_len,
                                          padding='post',
                                          value=self.answer_padding_idx),
            'samples_id': np.array(dev_cache_nopad['samples_id'])
        }

        logger.info('export to hdf5 file...')
        self._export_squad_hdf5()

        logger.info('finished.')


def dict2array(data_doc):
    """
    transform dict to numpy array
    :param data_doc: [{'token': [], 'pos': [], 'ent': [], 'em': [], 'em_lemma': [], 'right_space': []]
    :return:
    """
    data = {'token': [], 'pos': [], 'ent': [], 'em': [], 'em_lemma': [], 'right_space': []}
    max_len = 0

    for ele in data_doc:
        assert ele.keys() == data.keys()

        if len(ele['token']) > max_len:
            max_len = len(ele['token'])

        for k in ele.keys():
            if len(ele[k]) > 0:
                data[k].append(ele[k])

    for k in data.keys():
        if len(data[k]) > 0:
            data[k] = pad_sequences(data[k],
                                    maxlen=max_len,
                                    padding='post',
                                    value=PreprocessData.padding_idx)

    return data
