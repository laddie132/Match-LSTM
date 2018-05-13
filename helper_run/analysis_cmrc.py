# -*- coding: utf-8 -*-

import json
import pandas as pd
import matplotlib.pyplot as plt
import jieba


def analysis_dataset(json_file):

    with open(json_file) as f:
        dataset = json.load(f)

    print('context size:', len(dataset))

    qas_size = []   # to debug show
    context_len = []
    query_len = []
    ans_size = []   # to debug show
    ans_len = []

    for sample in dataset:
        context = sample['context_text']
        context = list(jieba.cut(context))

        qas = sample['qas']
        qas_size.append(len(qas))

        for qa in qas:
            context_len.append(len(context))

            query = qa['query_text']
            answer = qa['answers']

            query = list(jieba.cut(query))
            query_len.append(len(query))
            ans_size.append(len(answer))

            min_ans_len = min([len(list(jieba.cut(str(a)))) for a in answer])
            ans_len.append(min_ans_len)

    print('qa size:', sum(qas_size))
    print('max answer size:', max(ans_size))
    print('min answer size:', min(ans_size))
    print('max qas size:', max(qas_size))
    print('min qas size:', min(qas_size))

    context_len_ser = pd.Series(data=context_len)
    context_len_cnt = context_len_ser.value_counts().to_frame(name='cnt')
    context_len_cnt['length'] = context_len_cnt.index

    context_len_cnt.plot.scatter('length', 'cnt')
    plt.xlabel('context length')

    query_len_ser = pd.Series(data=query_len)
    query_len_cnt = query_len_ser.value_counts().to_frame(name='cnt')
    query_len_cnt['length'] = query_len_cnt.index

    query_len_cnt.plot.scatter('length', 'cnt')
    plt.xlabel('query length')

    ans_len_ser = pd.Series(data=ans_len)
    ans_len_cnt = ans_len_ser.value_counts().to_frame(name='cnt')
    ans_len_cnt['length'] = ans_len_cnt.index

    ans_len_cnt.plot.scatter('length', 'cnt')
    plt.xlabel('answer length')


if __name__ == '__main__':
    # analysis_dataset('/Users/han/cmrc/dataset/cmrc2018_train.json')
    analysis_dataset('/Users/han/cmrc/dataset/cmrc2018_dev.json')
    plt.show()