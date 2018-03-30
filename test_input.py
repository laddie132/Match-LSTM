#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import torch
import logging
import nltk
import numpy as np
from dataset.squad_dataset import SquadDataset
from models.match_lstm import MatchLSTMModel
from utils.load_config import init_logging, read_config
from utils.utils import to_long_variable

init_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info('------------Match-LSTM TEST INPUT--------------')
    logger.info('loading config file...')
    global_config = read_config()

    # set random seed
    seed = global_config['model']['random_seed']
    enable_cuda = global_config['test']['enable_cuda']
    torch.manual_seed(seed)

    logger.info('reading squad dataset...')
    dataset = SquadDataset(global_config)

    logger.info('constructing model...')
    model = MatchLSTMModel(global_config)
    if enable_cuda:
        model = model.cuda()
    model.eval()  # let training = False, make sure right dropout

    # load model weight
    logger.info('loading model weight...')
    model_weight_path = global_config['data']['model_path']
    is_exist_model_weight = os.path.exists(model_weight_path)
    if not is_exist_model_weight:
        logger.info("not found model weight file on '%s'" % model_weight_path)
        exit(-1)

    weight = torch.load(model_weight_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weight, strict=False)

    # manual input qa
    context = "In 1870, Tesla moved to Karlovac, to attend school at the Higher Real Gymnasium, where he was " \
             "profoundly influenced by a math teacher Martin Sekuli\u0107.:32 The classes were held in German, " \
             "as it was a school within the Austro-Hungarian Military Frontier. Tesla was able to perform integral " \
             "calculus in his head, which prompted his teachers to believe that he was cheating. He finished a " \
             "four-year term in three years, graduating in 1873.:33 "
    question = "What language were classes held in at Tesla's school?"
    answer = "German"

    context_token = nltk.word_tokenize(context)
    question_token = nltk.word_tokenize(question)

    context_id = dataset.sentence_word2id(context_token)
    question_id = dataset.sentence_word2id(question_token)

    context_var = to_long_variable(context_id).view(1, -1)
    question_var = to_long_variable(question_id).view(1, -1)

    out_ans_prop = model.forward(context_var, question_var)
    out_ans_range = torch.max(out_ans_prop, 2)[1].data.numpy()

    start = out_ans_range[0][0]
    end = out_ans_range[0][1] + 1

    out_answer_id = context_id[start:end]
    out_answer = dataset.sentence_id2word(out_answer_id)

    logging.info('Predict Answer: ' + ' '.join(out_answer))


if __name__ == '__main__':
    main()