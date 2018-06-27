#/usr/bin/env python
# coding=utf-8
import os
import sys
import numpy as np
import pandas as pd
import jieba
import re 

sys.path.append('utils/')
import config
from pinyin import PinYin
str2pinyin = PinYin()
jieba.load_userdict(config.jieba_dict)
stopwords = [line.strip() for line in open(config.stopwords_path, 'r').readlines()]
stopwords = [w.decode('utf8') for w in stopwords]
# stopwords=[]
#if config.cut_char_level:
stopwords = [u'？', u'。', u'，',]

use_pinyin =False


def clean_str(x):
    punc = "蚂蚁  了 吗  的 ！？。，：；."
    punc = punc.decode("utf-8")
    return re.sub(ur"[%s]+" %punc, "", x)

def cut_single(x,cut_char_level):
    x = clean_str(x)
    res = []
    if cut_char_level:
        setence_seged = list(x.strip())
        #print(setence_seged)
    else:
        setence_seged = jieba.cut(x.strip(), HMM=True)
        #setence_seged=jieba.cut_for_search(x.strip(),HMM=Truec)

        #import jieba.analyse
        #setence_seged = jieba.analyse.extract_tags(x.strip(),topK=5,withWeight=False,allowPOS=['n','v'])
    
    for word in setence_seged:
        if word not in stopwords:
            my_word = word
            if use_pinyin:
                my_word = str2pinyin.char2pinyin(my_word)
            res.append(my_word)
    return res
def more(data,n):
    pass
def cut_word(path):

    data = pd.read_csv(path, sep='\t', encoding='utf8',#nrows=1000,
                       names=['id', 'q1', 'q2', 'label'])
    #data['id'] = range(len(data))#重塑id
    data['label'] = data['label'].fillna(0)
    data['q1_cut'] = data['q1'].map(lambda x: cut_single(x,cut_char_level=True))
    data['q2_cut'] = data['q2'].map(lambda x: cut_single(x,cut_char_level=True))

    data['q1_cut_word'] = data['q1'].map(lambda x: cut_single(x,cut_char_level=False))
    data['q2_cut_word'] = data['q2'].map(lambda x: cut_single(x,cut_char_level=False))
    print('cut done')
    
    print(data.shape)
    return data
    
    
def read_cut(path):
    if not os.path.exists(config.data_cut_hdf):
        data = cut_word(path)
        data.to_hdf(config.data_cut_hdf, "data")
    data = pd.read_hdf(config.data_cut_hdf)
    return data
if __name__ == '__main__':
    path = config.origin_csv
    #read_data(path)
    cut_word(path)
