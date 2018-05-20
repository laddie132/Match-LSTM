# -*- coding: utf-8 -*-

import re
import logging
import torch
import numpy as np


logger = logging.getLogger(__name__)


class DocText:
    """
    define one sample text, like one context or one question
    """

    def __init__(self, nlp, text, config):
        doc = nlp(text)
        self.config = config
        self.token = []
        self.lemma = []
        self.pos = []
        self.ent = []
        self.em = []
        self.em_lemma = []

        self.right_space = []   # record whether the right side of every token is a white space

        for t in doc:
            if t.is_space:
                continue

            self.token.append(t.text)
            end_idx = t.idx + len(t.text)
            if end_idx < len(text) and text[end_idx] in Space.WHITE_SPACE:
                self.right_space.append(1)
            else:
                self.right_space.append(0)

            if config['use_em_lemma']:
                self.lemma.append(t.lemma_)
                self.em_lemma.append(0)

            if config['use_pos']:
                self.pos.append(t.tag_)  # also be t.pos_

            if config['use_ent']:
                self.ent.append(t.ent_type_)

            if config['use_em']:
                self.em.append(0)

    def __len__(self):
        return len(self.token)

    def update_em(self, doc_text2):
        """
        set the exact mach and exact match on lemma features
        :param doc_text2: the doc text to match
        :return:
        """
        for i in range(len(self.em)):
            if self.config['use_em'] and self.token[i] in doc_text2.token:
                self.em[i] = 1

            if self.config['use_em_lemma'] and self.lemma[i] in doc_text2.lemma:
                self.em_lemma[i] = 1

    def to_id(self, feature_dict):
        """
        transform raw text to feature vector representation.
        it's slow, only used for interactive mode.
        :param feature_dict: ['id2word', 'id2char' 'id2pos', 'id2ent']
        :return:
        """
        sen_id = []
        add_features = {}
        feature_dict = {k: v.tolist() for k, v in feature_dict.items()}
        seq_len = len(self.token)

        if self.config['use_pos']:
            add_features['pos'] = torch.zeros((seq_len, len(feature_dict['id2pos'])), dtype=torch.float)
        if self.config['use_ent']:
            add_features['ent'] = torch.zeros((seq_len, len(feature_dict['id2ent'])), dtype=torch.float)
        if self.config['use_em']:
            add_features['em'] = torch.tensor(self.em, dtype=torch.float).unsqueeze(-1)
        if self.config['use_em_lemma']:
            add_features['em_lemma'] = torch.tensor(self.em_lemma, dtype=torch.float).unsqueeze(-1)

        for i in range(seq_len):
            # word
            word = self.token[i]
            if word in feature_dict['id2word']:
                sen_id.append(feature_dict['id2word'].index(word))
            else:
                sen_id.append(0)   # id=0 means padding value in preprocess
                logger.warning("word '%s' out of vocabulary" % word)

            # pos
            if self.config['use_pos']:
                pos = self.pos[i]
                if pos in feature_dict['id2pos']:
                    add_features['pos'][i][feature_dict['id2pos'].index(pos)] = 1
                else:
                    logging.warning("pos '%s' out of vocabulary" % pos)

            # ent
            if self.config['use_ent']:
                ent = self.ent[i]
                if ent in feature_dict['id2ent']:
                    add_features['ent'][i][feature_dict['id2ent'].index(ent)] = 1
                else:
                    logging.warning("ent '%s' out of vocabulary" % ent)

        rtn_features = None
        if len(add_features) > 0:
            rtn_features = torch.cat(list(add_features.values()), dim=-1)

        rtn_sen_id = torch.tensor(sen_id, dtype=torch.long)

        return rtn_sen_id, rtn_features


class Space:
    WHITE_SPACE = ' \t\n\r\u00A0\u1680​\u180e\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a' \
                  '​​\u202f\u205f​\u3000\u2028\u2029'

    @staticmethod
    def is_white_space(c):
        return c in Space.WHITE_SPACE

    @staticmethod
    def remove_white_space(s):
        return re.sub('['+Space.WHITE_SPACE+']', '', s)
