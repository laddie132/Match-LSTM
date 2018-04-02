#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import torch
import logging
import torch.optim as optim
from dataset.squad_dataset import SquadDataset
from models.match_lstm import MatchLSTMModel
from models.loss import MyNLLLoss
from utils.load_config import init_logging, read_config
from utils.functions import to_variable
from utils.eval import eval_on_model


init_logging()
logger = logging.getLogger(__name__)


def main():
    logger.info('------------Match-LSTM Train--------------')
    logger.info('loading config file...')
    global_config = read_config()

    # set random seed
    seed = global_config['model']['random_seed']
    enable_cuda = global_config['train']['enable_cuda']
    torch.manual_seed(seed)

    if torch.cuda.is_available() and not enable_cuda:
        logger.warning("CUDA is avaliable, you can enable CUDA in config file")
    elif not torch.cuda.is_available() and enable_cuda:
        raise ValueError("CUDA is not abaliable, please unable CUDA in config file")

    logger.info('reading squad dataset...')
    dataset = SquadDataset(global_config)

    logger.info('constructing model...')
    model = MatchLSTMModel(global_config)
    criterion = MyNLLLoss()
    if enable_cuda:
        model = model.cuda()

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
        model.load_state_dict(weight, strict=False)

    # training arguments
    logger.info('start training...')
    train_batch_size = global_config['train']['batch_size']
    valid_batch_size = global_config['train']['valid_batch_size']

    batch_train_loader = dataset.get_dataloader_train(train_batch_size)
    batch_dev_loader = dataset.get_dataloader_dev(valid_batch_size)

    clip_grad_max = global_config['train']['clip_grad_norm']

    best_valid_f1 = None
    # every epoch
    for epoch in range(global_config['train']['epoch']):
        # train
        model.train()  # set training = True, make sure right dropout
        sum_loss = train_on_model(model=model,
                                  criterion=criterion,
                                  optimizer=optimizer,
                                  batch_data=batch_train_loader,
                                  epoch=epoch,
                                  clip_grad_max=clip_grad_max,
                                  enable_cuda=enable_cuda)
        logger.info('epoch=%d, sum_loss=%.5f' % (epoch, sum_loss))

        # evaluate
        model.eval()  # let training = False, make sure right dropout
        valid_score_em, valid_score_f1, valid_loss = eval_on_model(model=model,
                                                                   criterion=criterion,
                                                                   batch_data=batch_dev_loader,
                                                                   epoch=epoch,
                                                                   enable_cuda=enable_cuda)
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


def train_on_model(model, criterion, optimizer, batch_data, epoch, clip_grad_max, enable_cuda):
    """
    train on every batch
    :param model:
    :param criterion:
    :param batch_data:
    :param optimizer:
    :param epoch:
    :param clip_grad_max:
    :param enable_cuda:
    :return:
    """
    batch_cnt = len(batch_data)
    sum_loss = 0.
    for i, batch in enumerate(batch_data):
        optimizer.zero_grad()

        # forward
        bat_context, bat_question, bat_answer_range = list(map(lambda x: to_variable(x, enable_cuda), list(batch)))
        pred_answer_range, _ = model.forward(bat_context, bat_question)

        # get loss
        loss = criterion.forward(pred_answer_range, bat_answer_range)
        loss.backward()

        torch.nn.utils.clip_grad_norm(model.parameters(), clip_grad_max)        # fix gradient explosion
        optimizer.step()  # update parameters

        # logging
        batch_loss = loss.cpu().data.numpy()
        sum_loss += batch_loss * bat_answer_range.shape[0]

        logger.info('epoch=%d, batch=%d/%d, loss=%.5f' % (epoch, i, batch_cnt, batch_loss))

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
    main()