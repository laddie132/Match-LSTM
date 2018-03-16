# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import re
import numpy as np


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
    epoch_loss = []
    p = re.compile(r'.*epoch=\d*, sum_loss=(\d*\.\d*).*')
    for line in log_txt:
        result = re.findall(p, line)
        if len(result) == 0:
            continue
        sum_loss = float(result[0])
        epoch_loss.append(sum_loss)

    return epoch_loss


value_log = []
with open('../logs/2-debug.log') as f_log:
    log_lines = f_log.readlines()
    # value_log = log_lines[3220:]
    value_log = log_lines

epoch_loss = analysis_log_sum_loss(value_log)
for i, sum_loss in enumerate(epoch_loss):
    print('epoch=%d, sum_loss=%f' % (i, sum_loss))

# plot
x = range(len(epoch_loss))
y = epoch_loss

plt.plot(x, y, marker='o')

# my_x_ticks = np.arange(0, len(epoch_loss), 2)
# plt.xticks(my_x_ticks)

plt.grid()
plt.show()
