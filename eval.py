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
    dev_data = dataset.get_dev_data(enable_cuda)
    batch_context, batch_question, batch_answer_range = dev_data['context'], dev_data['question'], dev_data['answer_range']
    pred_answer_prop = model.forward(batch_context, batch_question)
    pred_answer_range = torch.max(pred_answer_prop, 2)[1]

    # calculate the mean em and f1 score
    logger.info('evaluating...')
    num_em = 0
    score_f1 = 0.
    batch_size = pred_answer_range.shape[0]
    for i in batch_size:
        if evaluate_em(pred_answer_range[i], dev_data['answer_range'][i]):
            num_em += 1
        score_f1 += evaluate_f1(batch_context[i], pred_answer_range[i], batch_answer_range[i])

    score_em = num_em * 1. / batch_size
    score_f1 /= batch_size

    logger.info("eval data size: %d" % batch_size)
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
        if y_true[(i * 2):(i * 2 + 2)] == y_pred:
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