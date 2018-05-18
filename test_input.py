#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import torch
import logging
import spacy
import numpy as np
import matplotlib.pyplot as plt
from models import *
from dataset.squad_dataset import SquadDataset
from dataset.preprocess_data import DocText
from utils.load_config import init_logging, read_config
from utils.functions import to_long_tensor, count_parameters, draw_heatmap_sea

init_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info('------------MODEL TEST INPUT--------------')
    logger.info('loading config file...')
    global_config = read_config()

    # set random seed
    seed = global_config['global']['random_seed']
    torch.manual_seed(seed)

    torch.set_grad_enabled(False)  # make sure all tensors below have require_grad=False

    logger.info('reading squad dataset...')
    dataset = SquadDataset(global_config)

    logger.info('constructing model...')
    model_choose = global_config['global']['model']
    dataset_h5_path = global_config['data']['dataset_h5']
    if model_choose == 'base':
        model = BaseModel(dataset_h5_path,
                          model_config=read_config('config/base_model.yaml'))
    elif model_choose == 'match-lstm':
        model = MatchLSTM(dataset_h5_path)
    elif model_choose == 'match-lstm+':
        model = MatchLSTMPlus(dataset_h5_path)
    elif model_choose == 'r-net':
        model = RNet(dataset_h5_path)
    elif model_choose == 'm-reader':
        model = MReader(dataset_h5_path)
    else:
        raise ValueError('model "%s" in config file not recoginized' % model_choose)

    model.eval()  # let training = False, make sure right dropout
    logging.info('model parameters count: %d' % count_parameters(model))

    # load model weight
    logger.info('loading model weight...')
    model_weight_path = global_config['data']['model_path']
    is_exist_model_weight = os.path.exists(model_weight_path)
    assert is_exist_model_weight, "not found model weight file on '%s'" % model_weight_path

    weight = torch.load(model_weight_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(weight, strict=False)

    # manual input qa
    context = "In 1870, Tesla moved to Karlovac, to attend school at the Higher Real Gymnasium, where he was " \
             "profoundly influenced by a math teacher Martin Sekuli\u0107.:32 The classes were held in German, " \
             "as it was a school within the Austro-Hungarian Military Frontier. Tesla was able to perform integral " \
             "calculus in his head, which prompted his teachers to believe that he was cheating. He finished a " \
             "four-year term in three years, graduating in 1873.:33 "
    question1 = "What language were classes held in at Tesla's school?"
    answer1 = ["German"]

    question2 = "Who was Tesla influenced by while in school?"
    answer2 = ["Martin Sekuli\u0107"]

    question3 = "Why did Tesla go to Karlovac?"
    answer3 = ["attend school at the Higher Real Gymnasium", 'to attend school']

    # change here to select questions
    question = question1
    answer = answer1[0]

    # preprocess
    nlp = spacy.load('en')
    context_doc = DocText(nlp, context, global_config['preprocess'])
    question_doc = DocText(nlp, question, global_config['preprocess'])
    context_doc.update_em(question_doc)
    question_doc.update_em(context_doc)

    context_token = context_doc.token
    question_token = question_doc.token

    context_id_char = to_long_tensor(dataset.sentence_char2id(context_token))
    question_id_char = to_long_tensor(dataset.sentence_char2id(question_token))

    context_id, context_f = context_doc.to_id(dataset.meta_data)
    question_id, question_f = question_doc.to_id(dataset.meta_data)

    bat_input = [context_id, question_id, context_id_char, question_id_char, context_f, question_f]
    bat_input = [x.unsqueeze(0) if x is not None else x for x in bat_input]

    out_ans_prop, out_ans_range, vis_param = model.forward(*bat_input)
    out_ans_range = out_ans_range.numpy()

    start = out_ans_range[0][0]
    end = out_ans_range[0][1] + 1

    out_answer_id = context_id[start:end]
    out_answer = dataset.sentence_id2word(out_answer_id)

    logging.info('Predict Answer: ' + ' '.join(out_answer))

    # to show on visdom
    s = 0
    e = 48

    x_left = vis_param['match']['left']['alpha'][0, :, s:e].numpy()
    x_right = vis_param['match']['right']['alpha'][0, :, s:e].numpy()

    x_left_gated = vis_param['match']['left']['gated'][0, :, s:e].numpy()
    x_right_gated = vis_param['match']['right']['gated'][0, :, s:e].numpy()

    draw_heatmap_sea(x_left,
                     xlabels=context_token[s:e],
                     ylabels=question_token,
                     answer=answer,
                     save_path='data/test-left.png',
                     bottom=0.45)
    draw_heatmap_sea(x_right,
                     xlabels=context_token[s:e],
                     ylabels=question_token,
                     answer=answer,
                     save_path='data/test-right.png',
                     bottom=0.45)

    enable_self_match = False
    if enable_self_match:
        x_self_left = vis_param['self']['left']['alpha'][0, s:e, s:e].numpy()
        x_self_right = vis_param['self']['right']['alpha'][0, s:e, s:e].numpy()

        draw_heatmap_sea(x_self_left,
                         xlabels=context_token[s:e],
                         ylabels=context_token[s:e],
                         answer=answer,
                         save_path='data/test-self-left.png',
                         inches=(11, 11),
                         bottom=0.2)
        draw_heatmap_sea(x_self_right,
                         xlabels=context_token[s:e],
                         ylabels=context_token[s:e],
                         answer=answer,
                         save_path='data/test-self-right.png',
                         inches=(11, 11),
                         bottom=0.2)
    # plt.show()


if __name__ == '__main__':
    main()