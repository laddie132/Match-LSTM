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
    logger.info('------------Match-LSTM Train--------------')
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

    logger.info('reading squad dataset...')
    dataset = SquadDataset(squad_h5_path=global_config['data']['dataset_h5'])

    logger.info('constructing model...')
    model = MatchLSTMModel(global_config)
    criterion = nn.NLLLoss()
    if enable_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # optimizer
    optimizer_choose = global_config['train']['optimizer']
    optimizer_lr = global_config['train']['learning_rate']
    optimizer_beta = (global_config['train']['adamax_beta1'], global_config['train']['adamax_beta2'])
    optimizer_param = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adamax(optimizer_param,
                             lr=optimizer_lr,
                             betas=optimizer_beta)

    if optimizer_choose == 'sgd':
        optimizer = optim.SGD(optimizer_param,
                              lr=optimizer_lr)
    elif optimizer_choose == 'adam':
        optimizer = optim.Adam(optimizer_param,
                               lr=optimizer_lr,
                               betas=optimizer_beta)
    elif optimizer_choose != "adamax":
        logger.error('optimizer "%s" in config file not recoginized, use default: adamax' % optimizer_choose)

    logger.info('start training...')
    batch_size = global_config['train']['batch_size']
    batch_list = dataset.get_batch_data(batch_size)

    # every epoch
    for epoch in range(global_config['train']['epoch']):
        sum_loss = 0.

        # every batch
        for i, batch in enumerate(batch_list):
            optimizer.zero_grad()

            # forward
            bat_context, bat_question, bat_answer_range = batch['context'], batch['question'], batch['answer_range']
            pred_answer_range = model.forward(bat_context, bat_question)

            # get loss
            loss = criterion(pred_answer_range, bat_answer_range)
            loss = torch.mean(loss)
            loss.backward()
            optimizer.step()    # update parameters

            # logging
            batch_loss = loss.cpu().data.numpy()
            sum_loss += batch_loss * batch_size

            logger.info('epoch=%d, batch=%d, sum_loss=%.5f, batch_loss=%.5f, lr=%.6f' % (
                epoch, i, sum_loss, batch_loss, optimizer_lr))

    torch.save(model.state_dict(), global_config['data']['model_path'])

    logger.info('finished.')


if __name__ == '__main__':
    main()