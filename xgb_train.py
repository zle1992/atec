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
from Feats import data_2id,load_hum_feats
from help import score, train_batch_generator, train_batch_generator3,train_test, get_X_Y_from_df
from CutWord import read_cut,more
path = config.origin_csv
print('load data')
data = read_cut(path)  #cut word
data = data_2id(data)  # 2id
label = data.label                  
data = load_hum_feats(data,config.train_feats)


data['label'] = label

train, dev = train_test(data)
x_train, x_dev = train[config.feats],dev[config.feats]

x_dev['cnn1']=np.load(config.model_dir + '/val_pred_cnn_0.531129240017.npz.npy')[:,1]

x_train['cnn1']= np.load(config.model_dir + '/train_pred_cnn_0.531129240017.npz.npy')[:len(x_train),1]

y_train, y_dev = train.label,dev.label
print('train ssshape', x_train.shape)
print('dev ssshape', x_dev.shape)


# Train model
X_train = x_train
Y_train = y_train
X_valid = x_dev
Y_valid = y_dev

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