#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import torch
import logging
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
    seed = global_config['test']['random_seed']
    enable_cuda = global_config['test']['enable_cuda']
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
    if enable_cuda:
        model = model.cuda()

    model_path = global_config['test']['model_path']
    start_epoch = global_config['test']['start_epoch']
    end_epoch = global_config['test']['end_epoch']

    for epoch in range(start_epoch, end_epoch):

        # load model weight
        logger.info('loading model weight...')
        weight_path = model_path + 'model-weight.pt-epoch' + str(epoch)
        is_exist_model_weight = os.path.exists(weight_path)
        if not is_exist_model_weight:
            logger.info("not found model weight file on '%s'" % weight_path)
            exit(-1)

        weight = torch.load(weight_path, map_location=lambda storage, loc: storage)
        if enable_cuda:
            weight = torch.load(weight_path, map_location=lambda storage, loc: storage.cuda())
        model.load_state_dict(weight, strict=False)

        # forward
        logger.info('forwarding...')
        batch_size = global_config['test']['batch_size']
        batch_dev_data = dataset.get_batch_dev(batch_size, enable_cuda)
        batch_dev_data = [batch_dev_data[0]]

        score_em, score_f1 = eval_on_model(model, batch_dev_data)
        logger.info("epoch=%d, ave_score_em=%.2f, ave_score_f1=%.2f" % (epoch, score_em, score_f1))

    logging.info('finished.')


def eval_on_model(model, batch_data):
    """
    evaluate on a specific trained model
    :param model: model with weight loaded
    :param batch_data: test data with batches
    :return: (em, f1)
    """
    dev_data_size = 0
    num_em = 0
    score_f1 = 0.
    for bnum, batch in enumerate(batch_data):
        bat_context, bat_question, bat_answer_range = batch['context'], batch['question'], batch['answer_range']
        tmp_ans_prop = model.forward(bat_context, bat_question)
        tmp_ans_range = torch.max(tmp_ans_prop, 2)[1]

        tmp_size = tmp_ans_range.shape[0]
        dev_data_size += tmp_size

        # calculate the mean em and f1 score
        for i in range(tmp_size):
            if evaluate_em(tmp_ans_range[i].cpu().data.numpy(), bat_answer_range[i].cpu().data.numpy()):
                num_em += 1
            score_f1 += evaluate_f1(bat_context[i].cpu().data.numpy(),
                                    tmp_ans_range[i].cpu().data.numpy(),
                                    bat_answer_range[i].cpu().data.numpy())
        logger.info('batch=%d/%d, cur_score_em=%.2f, cur_score_f1=%.2f' %
                    (bnum, len(batch_data), num_em * 1. / dev_data_size, score_f1 / dev_data_size))

    score_em = num_em * 1. / dev_data_size
    score_f1 /= dev_data_size

    logger.info("eval data size: %d" % dev_data_size)
    return score_em, score_f1


def evaluate_em(y_pred, y_true):
    """
    exact match score
    :param y_pred: (answer_len,)
    :param y_true: (condidate_answer_len,)
    :return: bool
    """
    answer_len = 2
    candidate_answer_size = int(len(y_true)/answer_len)

    for i in range(candidate_answer_size):
        if (y_true[(i * 2):(i * 2 + 2)] == y_pred).all():
            return True

    return False


def evaluate_f1(context_tokens, y_pred, y_true):
    """
    treat answer as bag of tokens, calculate F1 score
    :param context_tokens: context with word tokens
    :param y_pred: (answer_len,)
    :param y_true: (condidate_answer_len,)
    :return: float
    """
    answer_len = 2
    candidate_answer_size = int(len(y_true) / answer_len)

    pred_tokens = set(context_tokens[y_pred[0]:y_pred[1]])
    if len(pred_tokens) == 0:
        return 0

    all_f1 = []
    for i in range(candidate_answer_size):
        tmp_true = y_true[(i * 2):(i * 2 + 2)]
        if tmp_true[1] == 0:
            continue

        true_tokens = set(context_tokens[tmp_true[0]:tmp_true[1]])
        same_tokens = pred_tokens.intersection(true_tokens)

        precision = len(same_tokens) * 1. / len(pred_tokens)
        recall = len(same_tokens) * 1. / len(true_tokens)

        f1 = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        all_f1.append(f1)

    return max(all_f1)


if __name__ == '__main__':
    main()