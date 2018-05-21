#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
import logging
from dataset.preprocess_squad import PreprocessSquad

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
        bat_em = evaluate_em(tmp_ans_range, bat_answer_range)
        num_em += bat_em.sum().item()

        bat_f1 = evaluate_f1(tmp_ans_range, bat_answer_range)
        score_f1 += bat_f1.sum().item()

        if epoch is None:
            logger.info('test: batch=%d/%d, cur_score_em=%.2f, cur_score_f1=%.2f, batch_loss=%.5f' %
                        (bnum, batch_cnt, num_em*100.0 / dev_data_size, score_f1*100.0 / dev_data_size, batch_loss))
        else:
            logger.info('epoch=%d, batch=%d/%d, cur_score_em=%.2f, cur_score_f1=%.2f, batch_loss=%.5f' %
                        (epoch, bnum, batch_cnt, num_em*100.0 / dev_data_size, score_f1*100.0 / dev_data_size, batch_loss))

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
    :param y_pred: (batch, answer_len)
    :param y_true: (batch, condidate_answer_len)
    :return:
    """
    assert y_pred.shape[1] == 2

    batch_size, condidate_answer_len = y_true.shape
    candidate_answer_size = int(condidate_answer_len/2)

    candidate_em = []
    for i in range(candidate_answer_size):
        cur_em = y_true[:, (i * 2):(i * 2 + 2)] == y_pred
        cur_em = cur_em[:, 0] & cur_em[:, 1]
        candidate_em.append(cur_em.long())

    candidate_em = torch.stack(candidate_em, dim=-1)    # (batch, answer_num)
    batch_em, _ = torch.max(candidate_em, dim=-1)   # (batch,)

    return batch_em


def evaluate_f1(y_pred, y_true):
    """
    treat answer as bag of tokens, calculate F1 score
    :param context_tokens: context with word tokens
    :param y_pred: (batch, answer_len)
    :param y_true: (batch, condidate_answer_len)
    :return:
    """
    assert y_pred.shape[1] == 2

    batch_size, condidate_answer_len = y_true.shape
    candidate_answer_size = int(condidate_answer_len / 2)

    pred_len = (y_pred[:, 1] - y_pred[:, 0] + 1).type(torch.float)
    eps = 1e-6

    candidate_f1 = []
    for i in range(candidate_answer_size):
        cur_true = y_true[:, (i * 2):(i * 2 + 2)]

        same_left = torch.stack([cur_true[:, 0], y_pred[:, 0]], dim=1)
        same_left, _ = torch.max(same_left, dim=1)

        same_right = torch.stack([cur_true[:, 1], y_pred[:, 1]], dim=1)
        same_right, _ = torch.min(same_right, dim=1)

        same_len = same_right - same_left + 1  # (batch_size,)
        same_len = torch.stack([same_len, torch.zeros_like(same_len)], dim=1)
        same_len, _ = torch.max(same_len, dim=1)

        same_len = same_len.type(torch.float)
        true_len = (cur_true[:, 1] - cur_true[:, 0] + 1).type(torch.float)

        pre = same_len / (pred_len + eps)
        rec = same_len / (true_len + eps)

        f1 = 2 * pre * rec / (pre + rec + eps)
        candidate_f1.append(f1)
    candidate_f1 = torch.stack(candidate_f1, dim=-1)    # (batch, answer_num)
    bat_f1, _ = torch.max(candidate_f1, dim=-1)

    return bat_f1
