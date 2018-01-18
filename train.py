#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import yaml
import torch
import logging
import numpy as np
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from dataset.preprocess_data import PreprocessData
from dataset.squad_dataset import SquadDataset
from models.match_lstm import MatchLSTM
from utils.load_config import init_logging, read_config

torch.manual_seed(1)
init_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info('------------Match-LSTM Train--------------')
    logger.info('loading config file...')
    global_config = read_config()

    is_exist_dataset_h5 = os.path.exists(global_config['data']['dataset_h5'])
    is_exist_embedding_h5 = os.path.exists(global_config['data']['embedding_h5'])
    logger.info('%s dataset hdf5 file' % ("found" if is_exist_dataset_h5 else "not found"))
    logger.info('%s glove hdf5 file' % ("found" if is_exist_embedding_h5 else "not found"))

    if (not is_exist_dataset_h5) or (not is_exist_embedding_h5):
        logger.info('preprocess data...')
        PreprocessData(global_config)

    logger.info('reading squad dataset...')
    dataset = SquadDataset(squad_h5_path=global_config['data']['dataset_h5'])

    logger.info('constructing model...')
    model = MatchLSTM(global_config)
    criterion = nn.NLLLoss()

    # optimizer
    optimizer_choose = global_config['train']['optimizer']
    optimizer_lr = float(global_config['model']['learning_rate'])
    optimizer_beta = (float(global_config['model']['adamax_beta1']), float(global_config['model']['adamax_beta2']))
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
        raise ValueError('optimizer "%s" in config file not recoginized' % optimizer_choose)

    logger.info('start training...')
    batch_size = int(global_config['train']['batch_size'])
    # every epoch
    for epoch in range(int(global_config['train']['epoch'])):
        batch_gen = dataset.get_batch_gen(batch_size)
        num_batch = 0
        sum_loss = 0.

        # every batch
        for i in range(num_batch):
            optimizer.zero_grad()

            # forward
            bat_context, bat_question, bat_answer_range = batch_gen()
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

    with open(global_config['data']['model_path'], 'wb') as f:
        torch.save(model, f)

    logger.info('finished.')


if __name__ == '__main__':
    main()