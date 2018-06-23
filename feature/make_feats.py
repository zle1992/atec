#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
import pandas as pd
import numpy as np


sys.path.append('utils/')
import config
from CutWord import read_cut
from feats0 import magic1
from feats1 import feats1_gen,extract_features
def save_feats1(data):
    
    data = feats1_gen(data)
    data = extract_features(data)
    feats1_path = 'data/cache/feats/feats1_train.csv'
    feats1 = ['len_diff',
              'shingle_similarity_1',
              'shingle_similarity_2',
              'shingle_similarity_3',
              'common_words',
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

    data[['id'] + feats1].to_csv(feats1_path, index=False)


def save_feats0(data):
    
    data = magic1(data)
    feats0_path = 'data/cache/feats/feats0_train.csv'
    feats0 = ['q1_freq', 'q2_freq', 'freq_mean',
              'freq_cross', 'q1_freq_sq', 'q2_freq_sq']
    data[['id'] + feats0].to_csv(feats0_path, index=False)

def merge_feats():
    df0 = pd.read_csv('data/cache/feats/feats0_train.csv')
    df1 = pd.read_csv('data/cache/feats/feats1_train.csv')
    df = pd.merge(df0,df1,on='id')
    return df

def merge_test(data):
    data=  feats1_gen(data)
    data = extract_features(data)
    data = magic1(data)
    return data[config.feats]

if __name__ == '__main__':
    path = config.origin_csv
    data = read_cut(path)  # cut word
    #data = data_2id(data)  # 2id
    save_feats0(data)
    save_feats1(data)
