#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
##########################################################################
"""


Authors: zhangle11(zhangle11@baidu.com)
Date:    2018-04-27 12:12:56

"""
import pandas as pd 
import pickle
import yaml
config = yaml.load(open('config.yaml', 'r'))

with open(config['word_embed_dict'], "rb") as f:
    word_voc = pickle.load(f)


def read_df(trainfile):
    data = pd.read_csv(trainfile, sep='\t', header=None,
                       encoding='utf-8', names=['id', 'title', 'content', 'label'])
    data['label'] = data['label'].apply(
        lambda x: int(x == 'POSITIVE'))  # 人写的是1 机器写的是0
    return data


def padding_id(ids, padding_token=0, padding_length=None):
    if len(ids) > padding_length:
        return ids[:padding_length]
    else:
        return ids + [padding_token] * (padding_length - len(ids))


def word2id(contents):
    ''' contents  str
    '''
    contents = str(contents)
    contents = contents.split()

    ids = [word_voc[c] if c in word_voc else len(word_voc) for c in contents]

    return padding_id(ids, padding_token=0, padding_length=config['word_maxlen'])


def data2id():

    data = read_df(config['train_cut'])
    data['title'] = data.title.map(lambda x: word2id(x))
    data['content'] = data.content.map(lambda x: word2id(x))
    print(data.head(4))
    data.to_pickle(config['train_cut_id_dump'])

data2id()