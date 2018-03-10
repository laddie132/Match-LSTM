#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import os
import torch
import logging
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
    if enable_cuda:
        model = model.cuda()

    logger.info('loading model weight...')
    is_exist_model_weight = os.path.exists(global_config['data']['model_path'])
    if not is_exist_model_weight:
        logger.info("not found model weight file on '%s'" % global_config['data']['model_path'])
        exit(-1)
    weight = torch.load(global_config['data']['model_path'], map_location=lambda storage, loc: storage)
    if enable_cuda:
        weight = torch.load(global_config['data']['model_path'], map_location=lambda storage, loc: storage.cuda())
    model.load_state_dict(weight)

    # forward
    logger.info('forwarding...')

    batch_size = global_config['test']['batch_size']
    batch_dev_data = dataset.get_batch_dev(batch_size, enable_cuda)

    dev_data_size = 0
    num_em = 0
    score_f1 = 0.
    for bnum, batch in enumerate(batch_dev_data):
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
        logger.info('batch=%d' % bnum)

    score_em = num_em * 1. / dev_data_size
    score_f1 /= dev_data_size

    logger.info("eval data size: %d" % dev_data_size)
    logger.info("em: %.2f, f1: %.2f" % (score_em, score_f1))


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
        true_tokens = set(context_tokens[tmp_true[0]:tmp_true[1]])
        same_tokens = pred_tokens.union(true_tokens)

        precision = len(same_tokens) * 1. / len(pred_tokens)
        recall = len(same_tokens) * 1. / len(true_tokens)

        f1 = 2 * precision * recall / (precision + recall)
        all_f1.append(f1)

    return max(all_f1)


if __name__ == '__main__':
    main()