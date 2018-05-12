#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = 'han'

import torch
import torch.nn.functional as F


class MyNLLLoss(torch.nn.modules.loss._Loss):
    """
    a standard negative log likelihood loss. It is useful to train a classification
    problem with `C` classes.

    Shape:
        - y_pred: (batch, answer_len, prob)
        - y_true: (batch, answer_len)
        - output: loss
    """
    def __init__(self):
        super(MyNLLLoss, self).__init__()

    def forward(self, y_pred, y_true):
        torch.nn.modules.loss._assert_no_grad(y_true)

        y_pred_log = torch.log(y_pred)
        loss = []
        for i in range(y_pred.shape[0]):
            tmp_loss = F.nll_loss(y_pred_log[i], y_true[i], reduce=False)
            one_loss = tmp_loss[0] + tmp_loss[1]
            loss.append(one_loss)

        loss_stack = torch.stack(loss)
        return torch.mean(loss_stack)


class RLLoss(torch.nn.modules.loss._Loss):
    """
    a reinforcement learning loss. f1 score

    Shape:
        - y_pred: (batch, answer_len)
        - y_true: (batch, answer_len)
        - output: loss
    """
    def __init__(self):
        super(RLLoss, self).__init__()

    def forward(self, y_pred, y_true, eps=1e-6):
        return NotImplementedError

        torch.nn.modules.loss._assert_no_grad(y_true)

        assert y_pred.shape[1] == 2

        same_left = torch.stack([y_true[:, 0], y_pred[:, 0]], dim=1)
        same_left, _ = torch.max(same_left, dim=1)

        same_right = torch.stack([y_true[:, 1], y_pred[:, 1]], dim=1)
        same_right, _ = torch.min(same_right, dim=1)

        same_len = same_right - same_left + 1   # (batch_size,)
        same_len = torch.stack([same_len, torch.zeros_like(same_len)], dim=1)
        same_len, _ = torch.max(same_len, dim=1)

        same_len = same_len.type(torch.float)

        pred_len = (y_pred[:, 1] - y_pred[:, 0] + 1).type(torch.float)
        true_len = (y_true[:, 1] - y_true[:, 0] + 1).type(torch.float)

        pre = same_len / (pred_len + eps)
        rec = same_len / (true_len + eps)

        f1 = 2 * pre * rec / (pre + rec + eps)

        return -torch.mean(f1)
