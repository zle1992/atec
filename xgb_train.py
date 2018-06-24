#!/usr/bin/python
# -*- coding: utf-8 -*-
import os

import sys
import numpy as np
import pandas as pd
import pickle
import time
from sklearn.model_selection import train_test_split
 

import xgboost as xgb

from time import time



sys.path.append('utils/')
sys.path.append('feature/')
import config
from Feats import data_2id,human_feats
from help import score, train_batch_generator, train_batch_generator3,train_test, get_X_Y_from_df
from CutWord import read_cut,more
path = config.origin_csv
print('load data')
data = read_cut(path)  #cut word
#data = data_2id(data)  # 2id
label = data.label                  
data = human_feats(data)

data['label'] = label


X_train, X_valid, Y_train, Y_valid = train_test_split(
data.drop(columns=['label']),data.label, test_size=0.2, random_state=42)
print(list(X_train))

# Set our parameters for xgboost
params = {}
params['objective'] = 'binary:logistic'
params['eval_metric'] = 'logloss'
params["max_depth"] = 8
params["silent"] = 1
params["seed"] = 1632

d_train = xgb.DMatrix(X_train, label=Y_train)
d_valid = xgb.DMatrix(X_valid, label=Y_valid)
watchlist = [(d_train, 'train'), (d_valid, 'valid')]

print('train first model....')
bst = xgb.train(params, d_train, 50000, watchlist,early_stopping_rounds=50, verbose_eval=10)


pred = bst.predict(d_valid)
pred = [int(x>0.5) for x in pred]
print(sum(pred))
print(sum(Y_valid))
pre, rec, f1 = score(Y_valid, pred)
print('p r f1 ', pre, rec, f1)

feature_score = bst.get_fscore()
feature_score = sorted(feature_score.items(), key=lambda x:x[1],reverse=True)
f = []
for k,v in feature_score:
    f.append(k)
print(f)
print(feature_score)