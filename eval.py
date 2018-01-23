#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from dataset.preprocess_data import PreprocessData
from dataset.squad_dataset import SquadDataset
from models.match_lstm import MatchLSTMModel
from utils.load_config import init_logging, read_config

init_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info('------------Match-LSTM Evaluate--------------')
    logger.info('loading config file...')
    global_config = read_config()

    # set random seed
    seed = global_config['train']['random_seed']
    enable_cuda = global_config['train']['enable_cuda']
    torch.manual_seed(seed)
    if enable_cuda:
        torch.cuda.manual_seed(seed)

    if torch.cuda.is_available() and not enable_cuda:
        logger.warning("CUDA is avaliable, you can enable CUDA in config file")
    elif not torch.cuda.is_available() and enable_cuda:
        logger.error("CUDA is not abaliable, please unable CUDA in config file")
        exit(-1)

    # handle dataset
    is_exist_dataset_h5 = os.path.exists(global_config['data']['dataset_h5'])
    is_exist_embedding_h5 = os.path.exists(global_config['data']['embedding_h5'])
    logger.info('%s dataset hdf5 file' % ("found" if is_exist_dataset_h5 else "not found"))
    logger.info('%s glove hdf5 file' % ("found" if is_exist_embedding_h5 else "not found"))

    if (not is_exist_dataset_h5) or (not is_exist_embedding_h5):
        logger.info('preprocess data...')
        preprocess = PreprocessData(global_config)
        preprocess.run()


if __name__ == '__main__':
    main()