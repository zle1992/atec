#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)
import sys
import keras
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
import numpy as np
import pandas as pd
from keras.activations import softmax
from keras import backend
# Model Load
sys.path.append('utils/')
sys.path.append('feature/')
import config
from Feats import data_2id, add_hum_feats
from CutWord import cut_word
from help import get_X_Y_from_df



sys.path.append('models')
from CNN import model_conv1D_, ABCNN2, dssm
from RNN import rnn_v1, Siamese_LSTM, my_rnn
from ESIM import esim, decomposable_attention
from ABCNN import ABCNN
from bimpm import bimpm
from MatchZoo import *
from help import *

from train import get_model

def load_data(in_path):
    print('load data')
    data = cut_word(in_path)
    data = data_2id(data)  # 2id
    data = add_hum_feats(data,config.test_featdires) #生成特征并加入
    return data

def make_test_cv_data(X_dev, model_name, kfolds):
    test_df = pd.DataFrame()
    test_df['pred'] = 0
    test_df['label'] = -1
    S_test = np.zeros((X_dev[0].shape[0],1))
 
    for kf in range(1, kfolds + 1):
        print('kf: ', kf)
        bst_model_path = config.stack_path + \
        "dp_feats_%d_%s_%d.h5" % (len(config.feats), model_name,kf)

        model,lr = get_model(model_name)
        model.load_weights(bst_model_path)
        model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr), metrics=[Precision, Recall, F1, ])
        pred = model.predict(X_dev, batch_size=config.batch_size)

        S_test+= pred

    S_test/= kfolds
    test_df['pred'] = [i[0] for i in S_test]
    test_df.to_csv(config.stack_path+'test_%s.csv' % (model_name))
         
    return pred


def single_submit(X,model_name):
    bst_model_path = config.model_dir + \
        "dp_feats_%d_embed_%s.h5" % (len(config.feats), model_name)
    model,lr = get_model(model_name)
    model.load_weights(bst_model_path)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr), metrics=[Precision, Recall, F1,])
    test_pred = model.predict(X, batch_size=config.batch_size)
    return test_pred



def submit(in_path,out_path,model_name,cv=False):
    data = load_data(in_path)
    X, _ = get_X_Y_from_df(data, False, False)
    print('load model and predict')
    if not cv:
        test_pred = single_submit(X,model_name)
    else:
        test_pred = make_test_cv_data(X, model_name, kfolds=5)
    print('save submit file')
    data['label'] = [int(x > 0.5) for x in test_pred]
    data[['id', 'label']].to_csv(out_path, index=False, header=None, sep='\t')

if __name__ == '__main__':

    cv = False
    cv = True
    model_name = 'cnn'
    submit(sys.argv[1], sys.argv[2],model_name,cv)

