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

def moredata(data,random_state):
    q1 = data[data.label==0][['q1','q1_cut']].sample(frac=0.2,random_state=random_state)
    q2 = data[data.label==0][['q2','q2_cut']].sample(frac=0.2,random_state=random_state)


    data_new = pd.DataFrame()
    data_new['q1'] = q1.q1.values
    data_new['q1_cut'] =q1.q1_cut.values
    data_new['q2'] = q2.q2.values
    data_new['q2_cut'] =q2.q2_cut.values
    data_new['id'] = -1
    data_new['label'] = 0
    return data_new
def moredata2(data,random_state):
    
    q2 = data[data.label==0][['q2','q2_cut']].sample(frac=0.2,random_state=random_state)

    q1 = data[data.label==0][['q1','q1_cut']].sample(frac=0.2,random_state=random_state)


    data_new = pd.DataFrame()
    data_new['q1'] = q1.q1.values
    data_new['q1_cut'] =q1.q1_cut.values
    data_new['q2'] = q2.q2.values
    data_new['q2_cut'] =q2.q2_cut.values
    data_new['id'] = -1
    data_new['label'] = 0
    return data_new


def more(data,n):
    print('more_data-----')
    for i in range(n+1):
        if i==0:
            data1 = pd.DataFrame()
        else :
            data1= data1.append( moredata(data,random_state=i))
            data1= data1.append( moredata2(data,random_state=i))

    return data.append(data1) 
def cut_word(path,cut_char_level):

    data = pd.read_csv(path, sep='\t', encoding='utf8',#nrows=1000,
                       names=['id', 'q1', 'q2', 'label'])
    #data['id'] = range(len(data))#重塑id
    data['label'] = data['label'].fillna(0)
    data['q1_cut'] = data['q1'].map(lambda x: cut_single(x,cut_char_level))
    data['q2_cut'] = data['q2'].map(lambda x: cut_single(x,cut_char_level))
    print('cut done')
    
    print(data.shape)
    return data
    
    
def read_cut(path):
    if not os.path.exists(config.data_cut_hdf):
        data = cut_word(path,config.cut_char_level)
        data.to_hdf(config.data_cut_hdf, "data")
    data = pd.read_hdf(config.data_cut_hdf)
    return data
if __name__ == '__main__':
    path = config.origin_csv
    #read_data(path)
    cut_word(path)
