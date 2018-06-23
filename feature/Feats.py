#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
import numpy as np
from collections import defaultdict


sys.path.append('utils/')

import config
from w2v import load_my_train_w2v, load_pre_train_w2v
from CutWord import cut_word, more

from feats1 import feats1_gen, extract_features
from feats0 import magic1
from feats3 import mytfidf
if config.use_pre_train:
    vocab, embed_weights = load_pre_train_w2v(config.origin_csv)
else:
    vocab, embed_weights = load_my_train_w2v(config.origin_csv)


def padding_id(ids, padding_token=0, padding_length=None):
    if len(ids) > padding_length:
        return ids[:padding_length]
    else:
        return ids + [padding_token] * (padding_length - len(ids))


def word2id(contents, word_voc):
    ''' contents  list
    '''
#     contents = str(contents)
#     contents = contents.split()

    ids = [word_voc[c] if c in word_voc else len(word_voc) for c in contents]

    return padding_id(ids, padding_token=0, padding_length=config.word_maxlen)


def data_2id(data):
    data['q1_cut_id'] = data['q1_cut'].map(lambda x: word2id(x, vocab))
    data['q2_cut_id'] = data['q2_cut'].map(lambda x: word2id(x, vocab))
    return data


feats0 = ['q1_freq', 'q2_freq', 'freq_mean',
          'freq_cross', 'q1_freq_sq', 'q2_freq_sq']

feats1 = ['len_diff',
          'shingle_similarity_1',
          'shingle_similarity_2',
          'shingle_similarity_3',
          ]
feats2 = ['common_words',
          'cwc_min',
          'cwc_max',
          'csc_min',
          'csc_max',
          'ctc_min',
          'ctc_max',
          'last_word_eq',
          'first_word_eq',
          'abs_len_diff',
          'mean_len',
          'token_set_ratio',
          'token_sort_ratio',
          'fuzz_ratio',
          'fuzz_partial_ratio',
          'longest_substr_ratio']


def save_feats(data, fun, feats_list, path):
    data = fun(data)
    data = data[feats_list]
    data.to_csv(path, index=False)
    return data


def merge_feats(data, featdirs):
    fun_list= [magic1,feats1_gen,extract_features]
    if not os.path.exists(featdirs[2]):
        df0 = save_feats(data=data, fun=magic1,
                         feats_list=feats0, path=featdirs[0])

        df1 = save_feats(data=data, fun=feats1_gen,
                         feats_list=feats1, path=featdirs[1])

        df2 = save_feats(data=data, fun=extract_features,
                         feats_list=feats2, path=featdirs[2])

        

    else:
        # pass
        df0 = pd.read_csv(featdirs[0])
        df1 = pd.read_csv(featdirs[1])
        df2 = pd.read_csv(featdirs[2])
        #df3 = pd.read_csv(featdirs[3])
    df = pd.concat([df0, df1, df2], axis=1)
    print(list(df))
    return df


def add_hum_feats(data, featsdirs):
    print(featsdirs)
    if config.nofeats:
        data['magic_feat'] = list(np.zeros((len(data), 2)))
    else:
        df = merge_feats(data, featsdirs)
        data['magic_feat'] = list(df[config.feats].values)
    return data


# if __name__ == '__main__':
#     path = config.origin_csv
#     data = read_cut(path)  # cut word
#     data = data_2id(data)  # 2id
#     save_feats0(data)
#     #save_feats1(data)
