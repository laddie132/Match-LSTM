#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import sys

sys.path.append(os.getcwd())

import argparse
import torch
import logging
from collections import OrderedDict
from models import *
from utils.load_config import init_logging, read_config

init_logging()
logger = logging.getLogger(__name__)


def transform(pre_model_path, tar_model_path, cur_model):
    pre_weight = torch.load(pre_model_path, map_location=lambda storage, loc: storage)
    pre_keys = pre_weight.keys()
    pre_value = pre_weight.values()

    cur_weight = cur_model.state_dict()
    del cur_weight['embedding.embedding_layer.weight']
    cur_keys = cur_weight.keys()

    assert len(pre_keys) == len(cur_keys)
    logging.info('pre-keys: ' + str(pre_keys))
    logging.info('cur-keys: ' + str(cur_keys))

    new_weight = OrderedDict(zip(cur_keys, pre_value))
    torch.save(new_weight, tar_model_path)


def main(pre_model_path, tar_model_path):
    logger.info('loading config file...')
    global_config = read_config()

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

    logging.info("transforming model from '%s' to '%s'..." % (pre_model_path, tar_model_path))
    transform(pre_model_path, tar_model_path, model)

    logging.info('finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="transform a old model weight to the newest network")
    parser.add_argument('--input', '-i', required=True, nargs=1, dest='pre_weight')
    parser.add_argument('--output', '-o', required=True, nargs=1, dest='tar_weight')

    args = parser.parse_args()
    main(args.pre_weight[0], args.tar_weight[0])
