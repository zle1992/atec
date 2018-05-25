#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
##########################################################################
"""


Authors: zhangle11(zhangle11@baidu.com)
Date:    2018-04-27 10:49:08

"""
import os
import sys


import pandas as pd
#from joblib import Parallel, delayed
import jieba

import yaml
config = yaml.load(open('config.yaml', 'r'))


def read_df(trainfile):
    data = pd.read_csv(trainfile, sep='\\t', header=None, #nrows=60000,
                       encoding='utf-8', names=['id', 'q1', 'q2', 'label'])
    print(data.head())
    return data


def word_cut(df):
    with open(config['train_cut'], 'a+') as f:
        line = '\t'.join([df[0],' '.join(jieba.cut(df[1])) ,' '.join(jieba.cut(df[2])),df[3]])   
        f.writelines(line)
        f.writelines('\n')


def applyParallel(content, func, n_thread):
    with Parallel(n_jobs=n_thread) as parallel:
        parallel(delayed(func)(c) for c in content)


def main( trainfile):
    overwrite = True
    if overwrite:
        if os.path.exists(config['train_cut']):
            os.remove(config['train_cut'])

    trainfile = 'data/atec_nlp_sim_train.csv'
    df = read_df(trainfile)
    content = df.values
    #applyParallel(content, word_cut, 22)
if __name__ == '__main__':
    main()
