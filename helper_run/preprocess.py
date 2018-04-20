#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import sys

sys.path.append(os.getcwd())

import logging
from dataset.preprocess_data import PreprocessData
from utils.load_config import init_logging, read_config

init_logging()
logger = logging.getLogger(__name__)


def preprocess():
    logger.info('------------Preprocess SQuAD dataset--------------')
    logger.info('loading config file...')
    global_config = read_config()

    logger.info('preprocess data...')
    pdata = PreprocessData(global_config)
    pdata.run()


if __name__ == '__main__':
    preprocess()