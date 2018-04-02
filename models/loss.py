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