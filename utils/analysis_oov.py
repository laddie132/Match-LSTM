# -*- coding: utf-8 -*-

# 验证目前分词效果与原作者代码分词一致性，即现有分词结果oov是否均出现在原作者代码分词结果中


standard_words = set()
squad_tokenized_path = '/Users/han/MatchLSTM-PyTorch-master/tokenized_squad_v1.1.2/'
squad_tokenized_files = ['train-v1.1-story.txt', 'train-v1.1-question.txt',
                         'valid-v1.1-story.txt', 'valid-v1.1-question.txt',
                         'dev-v1.1-story.txt', 'dev-v1.1-question.txt']

line_num = 0
for sfile in squad_tokenized_files:
    with open(squad_tokenized_path + sfile) as sf:
        word_lines = sf.readlines()
        for line in word_lines:
            words = line.strip().split(' ')
            standard_words = standard_words.union(set(words))

            line_num += 1
            if line_num % 10000 == 0:
                print('processing line No.%d' % line_num)

oov_path = '../data/oov_word.txt'
with open(oov_path) as oov_f:
    oov_lines = oov_f.readlines()
    words = map(lambda line: line.strip(), oov_lines)
    oov_words = set(words)


with open('../data/standard_words.txt', 'w') as sd_f:
    sd_words_list = list(standard_words)
    sd_words = map(lambda line: line + '\n', sd_words_list)

    sd_f.writelines(sd_words)

print('squad words num:', len(standard_words))
print('oov words num:', len(oov_words))
print('the same num:', len(oov_words.intersection(standard_words)))
print('difference:\n', oov_words.difference(standard_words))