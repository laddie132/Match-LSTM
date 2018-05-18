#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
import logging
from dataset.preprocess_data import PreprocessData

logger = logging.getLogger(__name__)


def eval_on_model(model, criterion, batch_data, epoch, device):
    """
    evaluate on a specific trained model
    :param model: model with weight loaded
    :param criterion:
    :param batch_data: test data with batches
    :param epoch:
    :param device:
    :return: (em, f1, sum_loss)
    """
    batch_cnt = len(batch_data)
    dev_data_size = 0
    num_em = 0
    score_f1 = 0.
    sum_loss = 0.

    for bnum, batch in enumerate(batch_data):

        # batch data
        batch = [x.to(device) if x is not None else x for x in batch]
        bat_context = batch[0]
        bat_answer_range = batch[-1]

        # forward
        batch_input = batch[:len(batch) - 1]
        tmp_ans_prop, tmp_ans_range, _ = model.forward(*batch_input)

        tmp_size = bat_answer_range.shape[0]
        dev_data_size += tmp_size

        # get loss
        batch_loss = criterion.forward(tmp_ans_prop, bat_answer_range[:, 0:2])
        sum_loss += batch_loss.item() * tmp_size

        # calculate the mean em and f1 score
        for i in range(tmp_size):
            if evaluate_em(tmp_ans_range[i].cpu().numpy(), bat_answer_range[i].cpu().numpy()):
                num_em += 1
            score_f1 += evaluate_f1(bat_context[i].cpu().numpy(),
                                    tmp_ans_range[i].cpu().numpy(),
                                    bat_answer_range[i].cpu().numpy())
        if epoch is None:
            logger.info('test: batch=%d/%d, cur_score_em=%.2f, cur_score_f1=%.2f, batch_loss=%.5f' %
                        (bnum, batch_cnt, num_em * 1. / dev_data_size, score_f1 / dev_data_size, batch_loss))
        else:
            logger.info('epoch=%d, batch=%d/%d, cur_score_em=%.2f, cur_score_f1=%.2f, batch_loss=%.5f' %
                        (epoch, bnum, batch_cnt, num_em * 1. / dev_data_size, score_f1 / dev_data_size, batch_loss))

        # manual release memory, todo: really effect?
        del bat_context, bat_answer_range, batch, batch_input
        del tmp_ans_prop, tmp_ans_range, batch_loss
        # torch.cuda.empty_cache()

    score_em = num_em * 100.0 / dev_data_size
    score_f1 = score_f1 * 100.0 / dev_data_size

    logger.info("eval data size: %d" % dev_data_size)
    return score_em, score_f1, sum_loss


# ---------------------------------------------------------------------------------
# Here is the two evaluate function modified from standard file 'evaluate-v1.1.py'.
# We just use it to show how model effect during training or evaluating.
# If you want the standard score, please use 'test.py' to output answer json file
# and then use 'evaluate-v1.1.py' to evaluate
# ---------------------------------------------------------------------------------

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

    pred_tokens = set(context_tokens[y_pred[0]:y_pred[1]+1])
    if len(pred_tokens) == 0:
        return 0

    all_f1 = []
    for i in range(candidate_answer_size):
        tmp_true = y_true[(i * 2):(i * 2 + 2)]
        if tmp_true[0] == PreprocessData.answer_padding_idx:
            continue

        true_tokens = set(context_tokens[tmp_true[0]:tmp_true[1]+1])
        same_tokens = pred_tokens.intersection(true_tokens)

        precision = len(same_tokens) * 1. / len(pred_tokens)
        recall = len(same_tokens) * 1. / len(true_tokens)

        f1 = 0
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        all_f1.append(f1)

    return max(all_f1)