#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import zipfile
import spacy
import json
import logging
from functools import reduce
from .doc_text import DocTextEn, Space
from .preprocess import Preprocess

logger = logging.getLogger(__name__)


class PreprocessSquad(Preprocess):
    """
    preprocess dataset and glove embedding to hdf5 files, for SQuAD dataset
    """

    def __init__(self, global_config):
        super(PreprocessSquad, self).__init__(global_config)

        self._nlp = spacy.load('en')
        self._nlp.remove_pipe('parser')
        if not any([self._use_em_lemma, self._use_pos, self._use_ent]):
            self._nlp.remove_pipe('tagger')
        if not self._use_ent:
            self._nlp.remove_pipe('ner')

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

            cur_context_doc = DocTextEn(self._nlp, cur_context, self.preprocess_config)
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

                cur_question_doc = DocTextEn(self._nlp, cur_question, self.preprocess_config)
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

                    pos_s, pos_e = self._find_ans_start_end(cur_context, cur_context_doc, cur_ans_text, cur_ans_start)
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

    def _find_ans_start_end(self, context_text, context_doc, answer_text, answer_start):
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

    def _handle_emb(self):
        """
        handle glove embeddings, restore embeddings with dictionary
        :return:
        """
        logger.info("read glove from text file %s" % self._emb_path)
        with zipfile.ZipFile(self._emb_path, 'r') as zf:
            if len(zf.namelist()) != 1:
                raise ValueError('glove file "%s" not recognized' % self._emb_path)

            glove_name = zf.namelist()[0]

            word_num = 0
            with zf.open(glove_name) as f:
                for line in f:
                    line_split = line.decode('utf-8').split(' ')
                    self._word2vec[line_split[0]] = [float(x) for x in line_split[1:]]

                    word_num += 1
                    if word_num % 10000 == 0:
                        logger.info('handle word No.%d' % word_num)
