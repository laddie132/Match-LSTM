# -*- coding: utf-8 -*-

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

        for t in doc:
            if t.is_space:
                continue

            self.token.append(t.text)

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
