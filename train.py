#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from dataset.preprocess_data import PreprocessData
from dataset.squad_dataset import SquadDataset
from models.match_lstm import MatchLSTMModel
from utils.load_config import init_logging, read_config
from utils.utils import MyNLLLoss
import eval


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
    logger.info('%s dataset hdf5 file' % ("found" if is_exist_dataset_h5 else "not found"))

    if not is_exist_dataset_h5:
        logger.info('preprocess data...')
        preprocess = PreprocessData(global_config)
        preprocess.run()

    logger.info('reading squad dataset...')
    dataset = SquadDataset(squad_h5_path=global_config['data']['dataset_h5'])

    logger.info('constructing model...')
    model = MatchLSTMModel(global_config)
    criterion = MyNLLLoss()
    if enable_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    logger.info('loading pre-trained weight...')
    weight_path = global_config['data']['model_path']
    if os.path.exists(global_config['data']['checkpoint_path']):
        with open(global_config['data']['checkpoint_path'], 'r') as checkpoint_f:
            weight_path = checkpoint_f.read()

    weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
    if enable_cuda:
        weight = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda())
    model.load_state_dict(weight)

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
    batch_list = dataset.get_batch_train(batch_size, enable_cuda)

    # every epoch
    for epoch in range(global_config['train']['epoch']):
        # every batch
        for i, batch in enumerate(batch_list):
            optimizer.zero_grad()

            # forward
            bat_context, bat_question, bat_answer_range = batch['context'], batch['question'], batch['answer_range']
            pred_answer_range = model.forward(bat_context, bat_question)

            # get loss
            loss = criterion(pred_answer_range, bat_answer_range)
            loss.backward()
            optimizer.step()    # update parameters

            # logging
            batch_loss = loss.cpu().data.numpy()

            logger.info('epoch=%d, batch=%d, loss=%.5f, lr=%.6f' % (
                epoch, i, batch_loss, optimizer_lr))

        # save model weight
        model_weight = model.state_dict()
        del model_weight['embedding.embedding_layer.weight']

        model_weight_path = global_config['data']['model_path'] + '-epoch%d' % epoch
        torch.save(model_weight, model_weight_path)
        with open(global_config['data']['checkpoint_path'], 'w') as checkpoint_f:
            checkpoint_f.write(model_weight_path)
    logger.info('finished.')

    # evaluate on the dev data
    eval.main()


if __name__ == '__main__':
    main()