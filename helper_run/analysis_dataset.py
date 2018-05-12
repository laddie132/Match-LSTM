#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import sys
import seaborn as sns

sys.path.append(os.getcwd())

import logging
from dataset.squad_dataset import SquadDataset
from utils.load_config import init_logging, read_config
import matplotlib.pyplot as plt

init_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info('------------Analysis SQuAD dataset--------------')
    logger.info('loading config file...')
    global_config = read_config()

    logger.info('reading squad dataset...')
    dataset = SquadDataset(global_config)

    train_context_len_cnt, train_context_len = dataset.gather_context_seq_len('train')
    dev_context_len_cnt, dev_context_len = dataset.gather_context_seq_len('dev')

    train_answer_len = dataset.gather_answer_seq_len('train', max_len=9)
    dev_answer_len = dataset.gather_answer_seq_len('dev', max_len=9)

    logging.info('train context length cnt: ' + str(train_context_len_cnt))
    logging.info('dev context length cnt: ' + str(dev_context_len_cnt))

    sns.set()
    train_context_len.plot.scatter('length', 'cnt', title='train')
    plt.xlabel('passage length')
    dev_context_len.plot.scatter('length', 'cnt', title='dev')
    plt.xlabel('passage length')

    train_answer_len.plot.line('length', 'cnt', marker='o', title='train')
    plt.xticks(range(len(train_answer_len['length'])), train_answer_len['length'])
    plt.xlabel('answer length')

    dev_answer_len.plot.line('length', 'cnt', marker='o', title='dev')
    plt.xticks(range(len(dev_answer_len['length'])), dev_answer_len['length'])
    plt.xlabel('answer length')

    plt.show()


if __name__ == '__main__':
    main()
