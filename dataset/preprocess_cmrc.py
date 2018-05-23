#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import bz2
import json
import logging
from .doc_text import DocTextCh, Space
from .preprocess import Preprocess

logger = logging.getLogger(__name__)


class PreprocessCMRC(Preprocess):
    """
    preprocess dataset and glove embedding to hdf5 files, for CMRC dataset
    """

    def __init__(self, global_config):
        super(PreprocessCMRC, self).__init__(global_config)

    def _read_json(self, path):
        """
        read json format file from raw squad text
        :param path: squad file path
        :return:
        """
        with open(path, 'r') as f:
            data = json.load(f)

        self._attr['dataset_name'] = 'CMRC'
        return data

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
            cur_context = question_grp['context_text']
            cur_qas = question_grp['qas']

            cur_context_doc = DocTextCh(cur_context, self.preprocess_config)
            if training and len(cur_context_doc) > self._ignore_max_len:  # some context token len too large
                continue

            if self._use_char:
                self._update_to_char(cur_context)

            cur_context_ids = self._doctext_to_id(cur_context_doc)

            # every question-answer
            for qa in cur_qas:
                cur_question = qa['query_text']

                if self._use_char:
                    self._update_to_char(cur_question)

                cur_question_doc = DocTextCh(cur_question, self.preprocess_config)
                cur_question_ids = self._doctext_to_id(cur_question_doc)

                # get em feature
                if self._use_em or self._use_em_lemma:
                    cur_context_doc.update_em(cur_question_doc)
                    cur_question_doc.update_em(cur_context_doc)

                # chinese has no lemma feature
                cur_context_ids['em'] = cur_context_doc.em
                cur_context_ids['em_lemma'] = cur_context_doc.em_lemma
                cur_question_ids['em'] = cur_question_doc.em
                cur_question_ids['em_lemma'] = cur_question_doc.em_lemma

                contexts_doc.append(cur_context_ids)
                questions_doc.append(cur_question_ids)
                samples_id.append(qa['query_id'])

                # find all the answer positions
                cur_answers = qa['answers']
                self._max_answer_len = max(self._max_answer_len, len(cur_answers) * 2)

                cur_ans_range_ids = [0 for i in range(len(cur_answers) * 2)]
                for idx, cur_ans in enumerate(cur_answers):
                    cur_ans = Space.remove_white_space(str(cur_ans))  # it has some numbers
                    try:
                        char_pos_s = Space.remove_white_space(cur_context).index(cur_ans)
                    except ValueError:
                        if sum(cur_ans_range_ids) == self.answer_padding_idx * len(cur_answers) * 2:
                            logger.error('Answer not in context' +
                                         "\nContext:" + cur_context +
                                         "\nQuestion:" + cur_question +
                                         "\nAnswer:" + cur_ans)
                        continue

                    pos_s, pos_e = self._find_ans_start_end(cur_context, cur_context_doc, cur_ans, char_pos_s)
                    if pos_e < pos_s:
                        logger.error("Answer start position can't bigger than end position." +
                                     "\nContext:" + cur_context +
                                     "\nQuestion:" + cur_question +
                                     "\nAnswer:" + cur_ans)
                        continue

                    gen_ans = ''.join(cur_context_doc.token[pos_s:(pos_e + 1)])
                    if cur_ans not in gen_ans:
                        logger.error("Answer position wrong." +
                                     "\nContext:" + cur_context +
                                     "\nQuestion:" + cur_question +
                                     "\nAnswer:" + cur_ans)
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

    def _find_ans_start_end(self, context_text, context_doc, answer_text, answer_start):
        # find answer start position
        cur_ans_len = len(answer_text)
        char_pos_e = answer_start + cur_ans_len - 1

        pos_s = 0
        pos_e = 0
        is_find_s = False

        # find start and end position
        tmp_len = 0
        for token_i, token in enumerate(context_doc.token):
            tmp_len += len(token)

            if not is_find_s and tmp_len - 1 >= answer_start:
                pos_s = token_i
                is_find_s = True
            if tmp_len - 1 >= char_pos_e:
                pos_e = token_i
                break

        return pos_s, pos_e

    def _handle_emb(self):
        """
        handle glove embeddings, restore embeddings with dictionary
        :return:
        """
        logger.debug("read embeddings from text file %s" % self._emb_path)
        if not os.path.exists(self._emb_path):
            raise ValueError('embeddings file "%s" not existed' % self._emb_path)

        word_num = 0
        embedding_size = 0
        embedding_num = 0

        # some too long word maybe error, decode after read bytes
        with bz2.open(self._emb_path, mode='rb') as f:
            for line_b in f:
                try:
                    line = line_b.decode('utf-8')
                except UnicodeDecodeError:
                    # line = line_b.decode('latin-1')
                    continue

                line_split = line.strip().split(' ')
                if word_num == 0:
                    embedding_num = int(line_split[0])
                    embedding_size = int(line_split[1])
                    logger.info('Embedding size: %d' % embedding_size)
                    logger.info('Embedding num: %d' % embedding_num)

                else:
                    self._word2vec[line_split[0]] = [float(x) for x in line_split[1:]]

                word_num += 1
                if word_num % 10000 == 0:
                    logger.info('handle word No.%d' % word_num)
