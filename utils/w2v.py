# -*- coding: utf-8 -*-
import os
import numpy as np
import argparse
import pandas as pd
import sys

import config
import process

import random


def read_words(data):
    words = list(data.q1_cut) + list(data.q2_cut)
    words_all = []
    for word in words:
        words_all.extend(word)
    words_set = list(set(words_all))
    words_set = [u'unknow'] + words_set
    words_set = words_set + [u'pos', u'eos', u'padding']
    words_set = np.array(words_set)
    return words_set


def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}
    with open(path, 'rb') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi, dim


def load_pre_train_embeddings(vocab, vectors,):
    vector_length = len(vocab)
    weights = np.zeros((vector_length, 300),   dtype='float32')
    cnt = 0
    # Normalization
    for index, word in enumerate(vocab):
        if word.encode('utf8') in vectors:
            weights[index] = vectors[word.encode('utf8')]
        else:
            weights[index] = np.random.random(size=weights.shape[1])
            cnt += 1
    print('vocab oov:{0}/{1}'.format(cnt, len(vocab)))
    return weights


def save_my_w2v(path):
    data = process.read_cut(path)
    vocab = read_words(data)
    # Read top n word vectors. Read all vectors when topn is 0
    vectors, iw, wi, dim = read_vectors('data/pre_w2v/sgns.zhihu.word', 0)

    m = load_pre_train_embeddings(vocab, vectors)
    np.save(config.word_embed_vocab, vocab)
    np.save(config.word_embed_weight, m)

if __name__ == '__main__':
    save_my_w2v(path=config.origin_csv)
