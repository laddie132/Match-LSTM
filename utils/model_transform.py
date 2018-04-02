#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
import logging
from collections import OrderedDict
from models.match_lstm import MatchLSTMModel
from utils.load_config import init_logging, read_config

init_logging()
logger = logging.getLogger(__name__)


def transform(pre_model_path, tar_model_path, cur_model):

    pre_weight = torch.load(pre_model_path, map_location=lambda storage, loc: storage)
    pre_value = pre_weight.values()

    cur_weight = cur_model.state_dict()
    del cur_weight['embedding.embedding_layer.weight']
    cur_keys = cur_weight.keys()

    new_weight = OrderedDict(zip(cur_keys, pre_value))

    torch.save(new_weight, tar_model_path)


def main(pre_model_path, tar_model_path):
    logger.info('loading config file...')
    global_config = read_config()

    logger.info('constructing model...')
    model = MatchLSTMModel(global_config)

    logging.info('transforming model...')
    transform(pre_model_path, tar_model_path, model)

    logging.info('finished.')


if __name__ == '__main__':
    main('data/model-bak/match-lstm.pt-epoch17-bak', 'data/match-lstm.pt-epoch17')