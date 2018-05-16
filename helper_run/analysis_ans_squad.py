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

sys.path.append(os.getcwd())
sns.set()


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate_with_wrong(dataset, predictions):
    f1 = exact_match = total = 0
    all_ans = []

    for article in dataset:
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]

                tmp_f1 = metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)
                tmp_em = metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += tmp_f1
                exact_match += tmp_em

                all_ans += [{'id': qa['id'],
                             'em': int(tmp_em),
                             'f1': tmp_f1 * 100.0,
                             'true_answer': ground_truths,
                             'predict_answer': prediction,
                             'context': context,
                             'question': qa['question']}]

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return {'exact_match': exact_match, 'f1': f1}, all_ans


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
    ans_len_df.loc[max_length] = {'length': '>=%d'%max_length, 'f1': max_f1, 'cnt': max_cnt}

    ans_len_df.plot.line('length', 'cnt', marker='o')
    plt.xticks(range(len(ans_len_df['length'])), ans_len_df['length'])
    plt.xlabel('answer length')

    ans_len_df.plot.line('length', 'f1', marker='o')
    plt.xticks(range(len(ans_len_df['length'])), ans_len_df['length'])
    plt.xlabel('answer length')


if __name__ == '__main__':
    expected_version = '1.1'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    parser.add_argument('out_ans_file', help='Output answer file')
    args = parser.parse_args()
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)

    score, all_ans = evaluate_with_wrong(dataset, predictions)
    print(json.dumps(score))
    print(json.dumps(ana_question_type_f1(all_ans)))
    ana_length_f1(all_ans)

    with open(args.out_ans_file, 'w') as f:
        json.dump(all_ans, f)

    plt.show()

