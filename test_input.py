#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import torch
import logging
import nltk
import numpy as np
import matplotlib.pyplot as plt
from dataset.h5_dataset import Dataset
from models import *
from utils.load_config import init_logging, read_config
from utils.functions import to_long_tensor, count_parameters, draw_heatmap_sea
from dataset.preprocess_cmrc import hanlp_segment

init_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info('------------Match-LSTM TEST INPUT--------------')
    logger.info('loading config file...')
    global_config = read_config('config/CMRC.yaml')

    # set random seed
    seed = global_config['global']['random_seed']
    torch.manual_seed(seed)

    torch.set_grad_enabled(False)  # make sure all tensors below have require_grad=False

    logger.info('reading squad dataset...')
    dataset = Dataset(global_config)

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

    if global_config['test']['is_english']:
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
    else:
        context = "《战国无双3》（）是由光荣和ω-force开发的战国无双系列的正统第三续作。本作以三大故事为主轴，分别是以武田信玄等人为主的《关东三国志》，织田信长等人为主的《战国三杰》，石田三成等人为主的《关原的年轻武者》，丰富游戏内的剧情。此部份专门介绍角色，欲知武器情报、奥义字或擅长攻击类型等，请至战国无双系列1.由于乡里大辅先生因故去世，不得不寻找其他声优接手。从猛将传 and Z开始。2.战国无双 编年史的原创男女主角亦有专属声优。此模式是任天堂游戏谜之村雨城改编的新增模式。本作中共有20张战场地图（不含村雨城），后来发行的猛将传再新增3张战场地图。但游戏内战役数量繁多，部分地图会有兼用的状况，战役虚实则是以光荣发行的2本「战国无双3 人物真书」内容为主，以下是相关介绍。（注：前方加☆者为猛将传新增关卡及地图。）合并本篇和猛将传的内容，村雨城模式剔除，战国史模式可直接游玩。主打两大模式「战史演武」&「争霸演武」。系列作品外传作品"
        question1 = "《战国无双3》是由哪两个公司合作开发的？"
        answer1 = ['光荣和ω-force']

        question2 = '男女主角亦有专属声优这一模式是由谁改编的？'
        answer2 = ['村雨城', '任天堂游戏谜之村雨城']

        question3 = '战国史模式主打哪两个模式？'
        answer3 = ['「战史演武」&「争霸演武」']

    # change here to select questions
    question = question1
    answer = answer1[0]

    # preprocess
    if global_config['test']['is_english']:
        context_token = nltk.word_tokenize(context)
        question_token = nltk.word_tokenize(question)
    else:
        context_token = hanlp_segment(context)
        question_token = hanlp_segment(question)

    context_array = np.array(context_token)

    context_id = dataset.sentence_word2id(context_token)
    question_id = dataset.sentence_word2id(question_token)

    context_id_char = dataset.sentence_char2id(context_token)
    question_id_char = dataset.sentence_char2id(question_token)

    context_var, question_var, context_var_char, question_var_char = [to_long_tensor(x).unsqueeze(0)
                                                                      for x in [context_id, question_id,
                                                                                context_id_char, question_id_char]]

    out_ans_prop, out_ans_range, vis_param = model.forward(context_var, question_var, context_var_char, question_var_char)
    out_ans_range = out_ans_range.cpu().data.numpy()

    start = out_ans_range[0][0]
    end = out_ans_range[0][1] + 1

    out_answer_id = context_id[start:end]
    out_answer = dataset.sentence_id2word(out_answer_id)

    logging.info('Predict Answer: ' + ' '.join(out_answer))

    # to show on visdom
    s = 0
    e = 48

    x_left = vis_param['match']['left']['alpha'][0, :, s:e].data.numpy()
    x_right = vis_param['match']['right']['alpha'][0, :, s:e].data.numpy()

    x_left_gated = vis_param['match']['left']['gated'][0, :, s:e].data.numpy()
    x_right_gated = vis_param['match']['right']['gated'][0, :, s:e].data.numpy()

    draw_heatmap_sea(x_left,
                     xlabels=context_token[s:e],
                     ylabels=question_token,
                     answer=answer,
                     save_path='cmrc_data/test-left.png',
                     bottom=0.45)
    draw_heatmap_sea(x_right,
                     xlabels=context_token[s:e],
                     ylabels=question_token,
                     answer=answer,
                     save_path='cmrc_data/test-right.png',
                     bottom=0.45)

    enable_self_match = False
    if enable_self_match:
        x_self_left = vis_param['self']['left']['alpha'][0, s:e, s:e].data.numpy()
        x_self_right = vis_param['self']['right']['alpha'][0, s:e, s:e].data.numpy()

        draw_heatmap_sea(x_self_left,
                         xlabels=context_token[s:e],
                         ylabels=context_token[s:e],
                         answer=answer,
                         save_path='cmrc_data/test-self-left.png',
                         inches=(11, 11),
                         bottom=0.2)
        draw_heatmap_sea(x_self_right,
                         xlabels=context_token[s:e],
                         ylabels=context_token[s:e],
                         answer=answer,
                         save_path='cmrc_data/test-self-right.png',
                         inches=(11, 11),
                         bottom=0.2)
    # plt.show()


if __name__ == '__main__':
    main()