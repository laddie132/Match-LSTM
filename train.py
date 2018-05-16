#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import torch
import logging
import argparse
import torch.optim as optim
from dataset.squad_dataset import SquadDataset
from models import *
from models.loss import MyNLLLoss, RLLoss
from utils.load_config import init_logging, read_config
from utils.eval import eval_on_model
from utils.functions import pop_dict_keys

logger = logging.getLogger(__name__)


def train(config_path):
    logger.info('------------Match-LSTM Train--------------')
    logger.info('loading config file...')
    global_config = read_config(config_path)

    # set random seed
    seed = global_config['global']['random_seed']
    torch.manual_seed(seed)

    enable_cuda = global_config['train']['enable_cuda']
    device = torch.device("cuda" if enable_cuda else "cpu")
    if torch.cuda.is_available() and not enable_cuda:
        logger.warning("CUDA is avaliable, you can enable CUDA in config file")
    elif not torch.cuda.is_available() and enable_cuda:
        raise ValueError("CUDA is not abaliable, please unable CUDA in config file")

    logger.info('reading squad dataset...')
    dataset = SquadDataset(global_config)

    logger.info('constructing model...')
    model_choose = global_config['global']['model']
    dataset_h5_path = global_config['data']['dataset_h5']
    if model_choose == 'base':
        model_config = read_config('config/base_model.yaml')
        model = BaseModel(dataset_h5_path,
                          model_config)
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

    model = model.to(device)
    criterion = MyNLLLoss()

    # optimizer
    optimizer_choose = global_config['train']['optimizer']
    optimizer_lr = global_config['train']['learning_rate']
    optimizer_param = filter(lambda p: p.requires_grad, model.parameters())

    if optimizer_choose == 'adamax':
        optimizer = optim.Adamax(optimizer_param)
    elif optimizer_choose == 'adadelta':
        optimizer = optim.Adadelta(optimizer_param)
    elif optimizer_choose == 'adam':
        optimizer = optim.Adam(optimizer_param)
    elif optimizer_choose == 'sgd':
        optimizer = optim.SGD(optimizer_param,
                              lr=optimizer_lr)
    else:
        raise ValueError('optimizer "%s" in config file not recoginized' % optimizer_choose)

    # check if exist model weight
    weight_path = global_config['data']['model_path']
    if os.path.exists(weight_path):
        logger.info('loading existing weight...')
        weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
        if enable_cuda:
            weight = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda())
        # weight = pop_dict_keys(weight, ['pointer', 'init_ptr_hidden'])  # partial initial weight
        model.load_state_dict(weight, strict=False)

    # training arguments
    logger.info('start training...')
    train_batch_size = global_config['train']['batch_size']
    valid_batch_size = global_config['train']['valid_batch_size']

    # batch_train_data = dataset.get_dataloader_train(train_batch_size)
    # batch_dev_data = dataset.get_dataloader_dev(valid_batch_size)
    batch_train_data = list(dataset.get_batch_train(train_batch_size))
    batch_dev_data = list(dataset.get_batch_dev(valid_batch_size))

    clip_grad_max = global_config['train']['clip_grad_norm']
    enable_char = False
    if model_choose in global_config['global']['use_char_model'] or (
            model_choose == 'base' and model_config['encoder']['enable_char']):
        enable_char = True

    best_valid_f1 = None
    # every epoch
    for epoch in range(global_config['train']['epoch']):
        # train
        model.train()  # set training = True, make sure right dropout
        sum_loss = train_on_model(model=model,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  batch_data=batch_train_data,
                                  epoch=epoch,
                                  clip_grad_max=clip_grad_max,
                                  device=device,
                                  enable_char=enable_char,
                                  batch_char_func=dataset.gen_batch_with_char)
        logger.info('epoch=%d, sum_loss=%.5f' % (epoch, sum_loss))

        # evaluate
        with torch.no_grad():
            model.eval()  # let training = False, make sure right dropout
            valid_score_em, valid_score_f1, valid_loss = eval_on_model(model=model,
                                                                       criterion=criterion,
                                                                       batch_data=batch_dev_data,
                                                                       epoch=epoch,
                                                                       device=device,
                                                                       enable_char=enable_char,
                                                                       batch_char_func=dataset.gen_batch_with_char)
        logger.info("epoch=%d, ave_score_em=%.2f, ave_score_f1=%.2f, sum_loss=%.5f" %
                    (epoch, valid_score_em, valid_score_f1, valid_loss))

        # save model when best f1 score
        if best_valid_f1 is None or valid_score_f1 > best_valid_f1:
            save_model(model,
                       epoch=epoch,
                       model_weight_path=global_config['data']['model_path'],
                       checkpoint_path=global_config['data']['checkpoint_path'])
            logger.info("saving model weight on epoch=%d" % epoch)
            best_valid_f1 = valid_score_f1

    logger.info('finished.')


def train_on_model(model, criterion, optimizer, batch_data, epoch, clip_grad_max, device, enable_char, batch_char_func):
    """
    train on every batch
    :param enable_char:
    :param batch_char_func:
    :param model:
    :param criterion:
    :param batch_data:
    :param optimizer:
    :param epoch:
    :param clip_grad_max:
    :param device:
    :return:
    """
    batch_cnt = len(batch_data)
    sum_loss = 0.
    for i, batch in enumerate(batch_data):
        optimizer.zero_grad()

        # batch data
        bat_context, bat_question, bat_context_char, bat_question_char, bat_answer_range = \
            batch_char_func(batch, enable_char=enable_char, device=device)

        # forward
        ans_range_prop, _, _ = model.forward(bat_context, bat_question, bat_context_char, bat_question_char)

        # get loss
        loss = criterion.forward(ans_range_prop, bat_answer_range)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_max)  # fix gradient explosion
        optimizer.step()  # update parameters

        # logging
        batch_loss = loss.item()
        sum_loss += batch_loss * bat_answer_range.shape[0]

        logger.info('epoch=%d, batch=%d/%d, loss=%.5f' % (epoch, i, batch_cnt, batch_loss))

        # manual release memory, todo: really effect?
        del bat_context, bat_question, bat_answer_range, bat_context_char, bat_question_char
        del ans_range_prop, loss
        # torch.cuda.empty_cache()

    return sum_loss


def save_model(model, epoch, model_weight_path, checkpoint_path):
    """
    save model weight without embedding
    :param model:
    :param epoch:
    :param model_weight_path:
    :param checkpoint_path:
    :return:
    """
    # save model weight
    model_weight = model.state_dict()
    del model_weight['embedding.embedding_layer.weight']

    torch.save(model_weight, model_weight_path)

    with open(checkpoint_path, 'w') as checkpoint_f:
        checkpoint_f.write('epoch=%d' % epoch)


if __name__ == '__main__':
    init_logging()

    parser = argparse.ArgumentParser(description="train on the model")
    parser.add_argument('--config', '-c', required=False, dest='config_path', default='config/global_config.yaml')
    args = parser.parse_args()

    train(args.config_path)
