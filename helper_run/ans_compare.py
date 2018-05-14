#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import argparse
import json
import seaborn as sns
import pandas as pd
import numpy as np

sys.path.append(os.getcwd())


def set_diff(s1, s2):
    a = set(s1)
    b = set(s2)

    diff1 = a.difference(b)
    same = a.intersection(b)
    diff2 = b.difference(a)

    return len(diff1), len(same), len(diff2)


def get_label(s):
    if '/' in s:
        s = s.split('/')[-1]
    if '.' in s:
        s = s.split('.')[0]

    return s


def compre_two(ans_file1, ans_file2):

    with open(ans_file1) as f:
        a = json.load(f)
    with open(ans_file2) as f:
        b = json.load(f)

    true_a = []
    wrong_a = []
    for ele in a:
        if ele['em']:
            true_a.append(ele['id'])
        if not ele['f1']:
            wrong_a.append(ele['id'])

    true_b = []
    wrong_b = []
    for ele in b:
        if not ele['f1']:
            wrong_b.append(ele['id'])
        if ele['em']:
            true_b.append(ele['id'])

    label_a = get_label(ans_file1)
    label_b = get_label(ans_file2)
    venn2([set(true_a), set(true_b)], set_labels=(label_a, label_b))

    diff = set(true_b).intersection(set(wrong_a))
    if len(diff) > 20:
        diff = list(diff)[:20]
    print('true in b, but wrong in a:')
    print(diff)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare two different answer of SQuAD')
    parser.add_argument('answer_file1', help='Answer file 1')
    parser.add_argument('answer_file2', help='Answer File 2')

    args = parser.parse_args()
    compre_two(args.answer_file1, args.answer_file2)
    plt.show()

