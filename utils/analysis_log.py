# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import re


TRAIN_SIZE = 87599
DEV_SIZE = 10570


def analysis_log_loss(log_txt):
    epoch = []
    loss = []
    p = re.compile(r'.*epoch=(\d*), sum_loss=(\d*\.\d*).*')
    for line in log_txt:
        result = re.findall(p, line)
        if len(result) == 0:
            continue

        epoch.append(int(result[0][0]))
        loss.append(float(result[0][1])/TRAIN_SIZE)

    return epoch, loss


def analysis_log_score(log_txt):
    epoch = []
    score_em = []
    score_f1 = []
    loss = []
    p = re.compile(r'.*epoch=(\d*), ave_score_em=(\d*\.\d*), ave_score_f1=(\d*\.\d*), sum_loss=(\d*\.\d*)$')
    for line in log_txt:
        result = re.findall(p, line)
        if len(result) == 0:
            continue

        epoch.append(int(result[0][0]))
        score_em.append(float(result[0][1]))
        score_f1.append(float(result[0][2]))
        loss.append(float(result[0][3])/DEV_SIZE)

    return epoch, score_em, score_f1, loss


def draw_loss(epoch, train_loss, eval_loss):
    for i, tl, el in zip(epoch, train_loss, eval_loss):
        print('epoch=%d, train_loss=%f, eval_loss=%f' % (i, tl, el))

    # plot
    x = epoch
    y1 = train_loss
    y2 = eval_loss

    plt.figure()
    plt.plot(x, y1, marker='o', color='b')
    plt.plot(x, y2, marker='^', color='r')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(labels=['train', 'eval'])
    plt.grid()


def draw_score(epoch, score_em, score_f1):
    for i, em, f1 in zip(epoch, score_em, score_f1):
        print('epoch=%d, score_em=%.2f, score_f1=%.2f' % (i, em, f1))

    # plot
    x = epoch
    y1 = score_em
    y2 = score_f1

    plt.figure()
    plt.plot(x, y1, marker='o', color='b')
    plt.plot(x, y2, marker='^', color='r')

    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend(labels=['em', 'f1'])
    plt.grid()


def main(log_path):
    with open(log_path) as f_log:
        log_lines = f_log.readlines()
        value_log = log_lines

    epoch, train_loss = analysis_log_loss(value_log)
    epoch2, score_em, score_f1, eval_loss = analysis_log_score(value_log)

    assert epoch == epoch2

    draw_loss(epoch, train_loss, eval_loss)
    draw_score(epoch, score_em, score_f1)

    plt.show()


if __name__ == '__main__':
    main('../logs/match-lstm.log')