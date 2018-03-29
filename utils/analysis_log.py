# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import re


def analysis_log_loss(log_txt):
    epoch_loss = []
    p = re.compile(r'.*epoch=(\d*), batch=\d*.\d*, loss=(\d*\.\d*).*')
    for line in log_txt:
        result = re.findall(p, line)
        if len(result) == 0:
            continue

        tmp_epoch_loss = result[0]
        epoch = int(tmp_epoch_loss[0])
        loss = float(tmp_epoch_loss[1])

        if len(epoch_loss) < epoch + 1:
            epoch_loss.append(0.)
        else:
            epoch_loss[epoch] += loss
    return epoch_loss


def analysis_log_sum_loss(log_txt):
    epoch = []
    loss = []
    p = re.compile(r'.*epoch=(\d*), sum_loss=(\d*\.\d*).*')
    for line in log_txt:
        result = re.findall(p, line)
        if len(result) == 0:
            continue

        epoch.append(int(result[0][0]))
        loss.append(float(result[0][1]))

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
        loss.append(float(result[0][3]))

    return epoch, score_em, score_f1, loss


def main_loss():
    with open('../logs/6-debug.log') as f_log:
        log_lines = f_log.readlines()
        value_log = log_lines

    epoch, epoch_loss = analysis_log_sum_loss(value_log)
    for i, sum_loss in zip(epoch, epoch_loss):
        print('epoch=%d, sum_loss=%f' % (i, sum_loss))

    # plot
    x = epoch
    y = epoch_loss

    plt.plot(x, y, marker='o')

    plt.xlabel('epoch')
    plt.ylabel('sum_loss')
    plt.grid()
    plt.show()


def main_score():
    with open('../logs/6-debug.log') as f_log:
        log_lines = f_log.readlines()
        value_log = log_lines

    epoch, score_em, score_f1, loss = analysis_log_score(value_log)
    for i, em, f1, ls in zip(epoch, score_em, score_f1, loss):
        print('epoch=%d, score_em=%.2f, score_f1=%.2f, sum_loss=%.5f' % (i, em, f1, ls))

    # plot
    x = epoch
    y1 = score_em
    y2 = score_f1
    y3 = loss

    plt.figure(1)
    plt.plot(x, y1, marker='o', color='b')
    plt.plot(x, y2, marker='^', color='r')

    plt.xlabel('epoch')
    plt.ylabel('score')
    plt.legend(labels=['em', 'f1'])
    plt.grid()

    plt.figure(2)
    plt.plot(x, y3, marker='^', color='g')

    plt.xlabel('epoch')
    plt.ylabel('sum_loss')

    plt.grid()
    plt.show()


if __name__ == '__main__':
    main_loss()
    main_score()