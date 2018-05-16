#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Modified from official evaluation script for v1.1 of the SQuAD dataset. """

import os
import sys
import matplotlib.pyplot as plt
from collections import Counter
import string
import re
import argparse
import json
import seaborn as sns
import pandas as pd
import numpy as np
import nltk

sys.path.append(os.getcwd())
sns.set()


# split Chinese with English
def mixed_segmentation(in_str, rm_punc=False):
    in_str = str(in_str).lower().strip()
    segs_out = []
    temp_str = ""
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    for char in in_str:
        if rm_punc and char in sp_char:
            continue
        if re.search(u'[\u4e00-\u9fa5]', char) or char in sp_char:
            if temp_str != "":
                ss = nltk.word_tokenize(temp_str)
                segs_out.extend(ss)
                temp_str = ""
            segs_out.append(char)
        else:
            temp_str += char

    # handling last part
    if temp_str != "":
        ss = nltk.word_tokenize(temp_str)
        segs_out.extend(ss)

    return segs_out


# remove punctuation
def remove_punctuation(in_str):
    in_str = str(in_str).lower().strip()
    sp_char = ['-', ':', '_', '*', '^', '/', '\\', '~', '`', '+', '=',
               '，', '。', '：', '？', '！', '“', '”', '；', '’', '《', '》', '……', '·', '、',
               '「', '」', '（', '）', '－', '～', '『', '』']
    out_segs = []
    for char in in_str:
        if char in sp_char:
            continue
        else:
            out_segs.append(char)
    return ''.join(out_segs)


# find longest common string
def find_lcs(s1, s2):
    m = [[0 for i in range(len(s2) + 1)] for j in range(len(s1) + 1)]
    mmax = 0
    p = 0
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i + 1][j + 1] = m[i][j] + 1
                if m[i + 1][j + 1] > mmax:
                    mmax = m[i + 1][j + 1]
                    p = i + 1
    return s1[p - mmax:p], mmax


def calc_f1_score(answers, prediction):
    f1_scores = []
    for ans in answers:
        ans_segs = mixed_segmentation(ans, rm_punc=True)
        prediction_segs = mixed_segmentation(prediction, rm_punc=True)
        lcs, lcs_len = find_lcs(ans_segs, prediction_segs)
        if lcs_len == 0:
            f1_scores.append(0)
            continue
        precision = 1.0 * lcs_len / len(prediction_segs)
        recall = 1.0 * lcs_len / len(ans_segs)
        f1 = (2 * precision * recall) / (precision + recall)
        f1_scores.append(f1)
    return max(f1_scores)


def calc_em_score(answers, prediction):
    em = 0
    for ans in answers:
        ans_ = remove_punctuation(ans)
        prediction_ = remove_punctuation(prediction)
        if ans_ == prediction_:
            em = 1
            break
    return em


def evaluate_with_wrong(ground_truth_file, prediction_file):
    f1 = 0
    em = 0
    total_count = 0
    skip_count = 0
    all_ans = []

    for instance in ground_truth_file:
        context_id = instance['context_id'].strip()
        context_text = instance['context_text'].strip()
        for qas in instance['qas']:
            total_count += 1
            query_id = qas['query_id'].strip()
            query_text = qas['query_text'].strip()
            answers = qas['answers']

            if query_id not in prediction_file:
                sys.stderr.write('Unanswered question: {}\n'.format(query_id))
                skip_count += 1
                continue

            prediction = str(prediction_file[query_id]).replace(' ', '')
            tmp_f1 = calc_f1_score(answers, prediction)
            f1 += tmp_f1
            tmp_em = calc_em_score(answers, prediction)
            em += tmp_em

            all_ans.append({'id': query_id,
                            'em': int(tmp_em),
                            'f1': tmp_f1 * 100.0,
                            'true_answer': answers,
                            'predict_answer': prediction,
                            'context': context_text,
                            'question': query_text})

    f1_score = 100.0 * f1 / total_count
    em_score = 100.0 * em / total_count
    return (f1_score, em_score, total_count, skip_count), all_ans


def ana_question_type_f1(all_ans):
    qtype_f1 = {'What': 0,
                'Who': 0,
                'How': 0,
                'When': 0,
                'Which': 0,
                'Where': 0,
                'Why': 0,
                'Other': 0}
    qtype_cnt = {'What': 0,
                 'Who': 0,
                 'How': 0,
                 'When': 0,
                 'Which': 0,
                 'Where': 0,
                 'Why': 0,
                 'Other': 0}

    for ele in all_ans:
        cur_type = 'Other'
        for t in qtype_f1.keys():
            if t in ele['question'] or t.lower() in ele['question']:
                cur_type = t
                break

        if cur_type in qtype_f1.keys():
            qtype_cnt[cur_type] += 1
            qtype_f1[cur_type] += ele['f1']
        else:
            qtype_cnt['Other'] += 1
            qtype_f1['Other'] += ele['f1']

    for key in qtype_f1.keys():
        qtype_f1[key] = qtype_f1[key] * 1.0 / qtype_cnt[key]

    tmp_data = np.array([list(qtype_f1.values()), list(qtype_cnt.values())]).transpose()
    qtype_df = pd.DataFrame(data=tmp_data, index=qtype_f1.keys(), columns=['f1', 'cnt'])
    qtype_df['type'] = qtype_f1.keys()

    # sns.set_style('whitegrid')
    # sns.barplot(x='type', y='f1', data=qtype_df, color='#5f88bc')
    qtype_df.plot.line(x='type', y='f1', marker='o')
    plt.xlabel('question type')
    plt.xticks(range(len(qtype_df['type'])), qtype_df['type'])

    qtype_df.plot.line(x='type', y='cnt', marker='o')
    plt.xlabel('question type')
    plt.xticks(range(len(qtype_df['type'])), qtype_df['type'])

    return qtype_f1, qtype_cnt


def ana_length_f1(all_ans):
    ct_len_f1 = {}
    ct_len_cnt = {}
    ans_len_f1 = {}
    ans_len_cnt = {}

    for ele in all_ans:
        tmp_ct_len = len(ele['context'].split())
        tmp_ans_len = min([len(x.split()) for x in ele['true_answer']])

        if tmp_ct_len not in ct_len_f1:
            ct_len_f1[tmp_ct_len] = 0
            ct_len_cnt[tmp_ct_len] = 1
        ct_len_f1[tmp_ct_len] += ele['f1']
        ct_len_cnt[tmp_ct_len] += 1

        if tmp_ans_len not in ans_len_f1:
            ans_len_f1[tmp_ans_len] = 0
            ans_len_cnt[tmp_ans_len] = 1
        ans_len_f1[tmp_ans_len] += ele['f1']
        ans_len_cnt[tmp_ans_len] += 1

    ct_len_data = np.array(list(ct_len_f1.values())).T
    ct_len_df = pd.DataFrame(data=ct_len_data, columns=['f1'], index=ct_len_f1.keys())
    ct_len_df['cnt'] = list(ct_len_cnt.values())
    ct_len_df['length'] = ct_len_df.index

    ans_len_data = np.array(list(ans_len_f1.values())).T
    ans_len_df = pd.DataFrame(data=ans_len_data, columns=['f1'], index=ans_len_f1.keys())
    ans_len_df['cnt'] = list(ans_len_cnt.values())
    ans_len_df['length'] = ans_len_df.index

    ct_len_df['f1'] = ct_len_df['f1'] * 1.0 / ct_len_df['cnt']
    ans_len_df['f1'] = ans_len_df['f1'] * 1.0 / ans_len_df['cnt']

    # f1 for passage and answer length
    ct_len_df.plot.scatter('length', 'f1')
    plt.xlabel('passage length')
    ans_len_df.plot.scatter('length', 'f1')
    plt.xlabel('answer length')

    # count for passage length
    ct_len_df.plot.scatter('length', 'cnt')
    plt.xlabel('passage length')

    # count, f1 for answer length
    max_length = 9
    max_f1 = ans_len_df[ans_len_df['length'] >= max_length]['f1'].mean()
    max_cnt = ans_len_df[ans_len_df['length'] >= max_length]['cnt'].sum()

    ans_len_df = ans_len_df[ans_len_df['length'] < max_length]
    ans_len_df = ans_len_df.sort_index()
    ans_len_df.loc[max_length] = {'length': '>=%d' % max_length, 'f1': max_f1, 'cnt': max_cnt}

    ans_len_df.plot.line('length', 'cnt', marker='o')
    plt.xticks(range(len(ans_len_df['length'])), ans_len_df['length'])
    plt.xlabel('answer length')

    ans_len_df.plot.line('length', 'f1', marker='o')
    plt.xticks(range(len(ans_len_df['length'])), ans_len_df['length'])
    plt.xlabel('answer length')


if __name__ == '__main__':
    ground_truth_file = json.load(open(sys.argv[1], 'rb'))
    prediction_file = json.load(open(sys.argv[2], 'rb'))
    ans_out_file = sys.argv[3]

    score, all_ans = evaluate_with_wrong(ground_truth_file, prediction_file)
    # print(json.dumps(ana_question_type_f1(all_ans)))
    # ana_length_f1(all_ans)

    F1, EM, TOTAL, SKIP = score
    AVG = (EM + F1) * 0.5
    print('AVG: {:.3f} F1: {:.3f} EM: {:.3f} TOTAL: {} SKIP: {} FILE: {}'.format(AVG, F1, EM, TOTAL, SKIP, sys.argv[2]))

    with open(ans_out_file, 'w') as f:
        json.dump(all_ans, f)

    plt.show()
