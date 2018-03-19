#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import re
import os
import torch
import logging
import torch.nn as nn
import torch.optim as optim
from dataset.squad_dataset import SquadDataset
from models.match_lstm import MatchLSTMModel
from utils.load_config import init_logging, read_config
from utils.utils import MyNLLLoss


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

    logger.info('reading squad dataset...')
    dataset = SquadDataset(global_config)

    logger.info('constructing model...')
    model = MatchLSTMModel(global_config)
    criterion = MyNLLLoss()
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

    logger.info('loading checkpoint...')
    weight_path = global_config['data']['model_path']
    if os.path.exists(global_config['data']['checkpoint_path']):
        with open(global_config['data']['checkpoint_path'], 'r') as checkpoint_f:
            weight_path = checkpoint_f.read().strip()

    start_epoch = 0
    if 'epoch' in weight_path:
        p = re.compile('.*epoch(\d*)')
        start_epoch = int(re.findall(p, weight_path)[0]) + 1

    if os.path.exists(weight_path):
        weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
        if enable_cuda:
            weight = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(weight, strict=False)

    logger.info('start training from epoch %d...' % start_epoch)
    batch_size = global_config['train']['batch_size']
    batch_list = dataset.get_batch_train(batch_size, enable_cuda)

    # every epoch
    last_loss = 0.
    for epoch in range(start_epoch, global_config['train']['epoch']):
        sum_loss = 0.
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
            sum_loss += batch_loss * batch_size

            logger.info('epoch=%d, batch=%d/%d, loss=%.5f, lr=%.6f' % (
                epoch, i, len(batch_list), batch_loss, optimizer_lr))
        logger.info('epoch=%d, sum_loss=%.5f' % (epoch, sum_loss))

        # adjust learning rate when loss up
        if last_loss != 0. and sum_loss > last_loss:
            optimizer_lr = global_config['train']['learning_rate_decay_ratio'] * optimizer_lr
            for param_grp in optimizer.param_groups:
                param_grp['lr'] = optimizer_lr
            logging.info('learning rate down to %f' % optimizer_lr)
        last_loss = sum_loss

        # save model weight
        model_weight = model.state_dict()
        del model_weight['embedding.embedding_layer.weight']

        model_weight_path = global_config['data']['model_path'] + '-epoch%d' % epoch
        torch.save(model_weight, model_weight_path)

        logger.info("saving model weight on '%s'" % model_weight_path)
        with open(global_config['data']['checkpoint_path'], 'w') as checkpoint_f:
            checkpoint_f.write(model_weight_path)
    logger.info('finished.')


if __name__ == '__main__':
    main()