# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import re
import numpy as np

value_log = []
with open('../logs/1-epoch20.log') as f_log:
    log_lines = f_log.readlines()
    value_log = log_lines[3220:]

epoch_loss = []

p = re.compile(r'.*epoch=(\d*), batch=\d*.\d*, loss=(\d*\.\d*).*')
for line in value_log:
    if '[train.py:102-main()] - INFO - epoch' in line:
        tmp_epoch_loss = re.findall(p, line)[0]
        epoch = int(tmp_epoch_loss[0])
        loss = float(tmp_epoch_loss[1])

        if len(epoch_loss) < epoch + 1:
            epoch_loss.append(0.)
        else:
            epoch_loss[epoch] += loss
epoch_loss = epoch_loss[:len(epoch_loss)-1]

x = range(len(epoch_loss))
y = epoch_loss

plt.plot(x, y, marker='o')

my_x_ticks = np.arange(0, len(epoch_loss), 2)
plt.xticks(my_x_ticks)

plt.grid()
plt.show()
