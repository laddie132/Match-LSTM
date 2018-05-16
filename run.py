#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import logging
import argparse
from train import train
from test import test
from utils.load_config import init_logging, read_config
from dataset.preprocess_data import PreprocessData

init_logging()
logger = logging.getLogger(__name__)


def preprocess(config_path):
    logger.info('------------Preprocess SQuAD dataset--------------')
    logger.info('loading config file...')
    global_config = read_config(config_path)

    logger.info('preprocess data...')
    pdata = PreprocessData(global_config)
    pdata.run()


parser = argparse.ArgumentParser(description="preprocess/train/test the model")
parser.add_argument('mode', help='preprocess or train or test')
parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
parser.add_argument('--output', '-o', required=False, dest='out_path')
args = parser.parse_args()


if args.mode == 'preprocess':
    preprocess(args.config_path)
elif args.mode == 'train':
    train(args.config_path)
elif args.mode == 'test':
    test(config_path=args.config_path, out_path=args.out_path)
else:
    raise ValueError('Unrecognized mode selected.')

