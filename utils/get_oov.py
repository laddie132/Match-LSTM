# -*- coding: utf-8 -*-

import os
import re


with open('../logs/oov.log', 'r') as oov_f:
    oov_lines = oov_f.readlines()

p = re.compile(r'.*OOV word (.*)')
oov_word = []
for line in oov_lines:
    result = re.findall(p, line.strip())
    if len(result) == 0:
        print(line)
        continue

    word = result[0]
    oov_word.append(word + '\n')

with open('../data/oov_word.txt', 'w') as oov_f:
    oov_f.writelines(oov_word)

print('finished.')