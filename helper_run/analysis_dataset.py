#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import sys

sys.path.append(os.getcwd())

import logging
from dataset.squad_dataset import SquadDataset
from utils.load_config import init_logging, read_config
import matplotlib.pyplot as plt

init_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info('------------Analysis SquAD dataset--------------')
    logger.info('loading config file...')
    global_config = read_config()

    logger.info('reading squad dataset...')
    dataset = SquadDataset(global_config)

    train_context_len_cnt, train_context_len = dataset.gather_context_seq_len('train')
    dev_context_len_cnt, dev_context_len = dataset.gather_context_seq_len('dev')

    logging.info('train context length cnt: ' + str(train_context_len_cnt))
    logging.info('dev context length cnt: ' + str(dev_context_len_cnt))

    train_context_len.plot.scatter('length', 'cnt', title='train')
    dev_context_len.plot.scatter('length', 'cnt', title='dev')

    plt.show()


if __name__ == '__main__':
    main()
