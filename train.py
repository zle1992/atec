#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow as tf
tfconfig = tf.ConfigProto()
tfconfig.gpu_options.allow_growth = True
session = tf.Session(config=tfconfig)
import sys
import numpy as np
import pandas as pd
import pickle
import keras
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer
from keras.backend.tensorflow_backend import set_session
import time
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
import keras.backend as K
from keras.callbacks import TensorBoard
from keras.callbacks import Callback
from keras.callbacks import TensorBoard
sys.path.append('models')
from CNN import cnn_v1, cnn_v2, model_conv1D_,Siamese_LSTM,ABCNN2
from ESIM import esim, decomposable_attention
from ABCNN import ABCNN
from bimpm import bimpm
from MatchZoo import *
sys.path.append('utils/')
sys.path.append('feature/')
import config
from Feats import data_2id,add_hum_feats
from help import score, train_batch_generator, train_batch_generator3,train_test, get_X_Y_from_df
from CutWord import read_cut,more

def load_data():
    path = config.origin_csv
    print('load data')
    data = read_cut(path)  #cut word
    data = data_2id(data)  # 2id
    data = add_hum_feats(data,config.train_feats) #生成特征并加入
    train, dev = train_test(data)
    x_train, y_train = get_X_Y_from_df(train, config.data_augment,False)
    x_dev, y_dev = get_X_Y_from_df(dev, False,False)
    print('train shape', x_train[0].shape)
    print('dev shape', x_dev[0].shape)
    return x_train, y_train,x_dev, y_dev

def train(x_train, y_train,x_dev, y_dev,model_name, model):
    for i in range(15):
        K.set_value(model.optimizer.lr, 0.005)
        if i == 9:
            K.set_value(model.optimizer.lr, 0.0001)

        model.fit_generator(
            train_batch_generator3(x_train, y_train, config.batch_size),
            epochs=1,
            steps_per_epoch=int(y_train.shape[0] / config.batch_size),
            validation_data=(x_dev, y_dev),
            class_weight={0: 1, 1: 4},
            callbacks=[TensorBoard(log_dir='data/log_dir'),EarlyStopping(monitor='val_loss',patience=0,verbose=0,mode='auto')],
            verbose=1,

        )
        pred = model.predict(x_dev, batch_size=config.batch_size)
        pred_train = model.predict(x_train, batch_size=config.batch_size)
        
        pre, rec, f1 = score(y_dev, pred)

        np.save(config.model_dir + "/val_pred_%s_%s.npz" %
                   (model_name, f1),np.array(pred))
        np.save(config.model_dir + "/train_pred_%s_%s.npz" %
                   (model_name, f1),np.array(pred_train))

        
        model.save(config.model_dir + "/dp_embed_%s_%s.h5" %
                   (model_name, f1))
        print('p r f1 ', pre, rec, f1)


def main(model_name):
    print('model name', model_name)
    path = config.origin_csv
    x_train, y_train,x_dev, y_dev = load_data()
    
    if model_name == 'bimpm':
        model = bimpm()
    if model_name == 'drmmt':
        model = drmm_tks(num_layer=3, hidden_sizes=[100,80,1],topk=20)

    if model_name == 'msrnn':
        model = MATCHSRNN()
    if model_name == 'arc2':
        model = arc2()
    if model_name == 'test':
        model = test()
    if model_name == 'cnn':

        model = model_conv1D_()
    if model_name == 'slstm':

        model = Siamese_LSTM()

    if model_name == 'esim':
        model = esim()

    if model_name == 'dam':
        model = decomposable_attention()
    if model_name == 'abcnn':

        model = ABCNN2(
            left_seq_len=config.word_maxlen, right_seq_len=config.word_maxlen, depth=3,
            nb_filter=100, filter_widths=[5,4,3],
            collect_sentence_representations=True, abcnn_1=True, abcnn_2=False,
            #mode="euclidean",
            mode="cos",
            #mode='dot'
        )

    train(x_train, y_train,x_dev, y_dev,model_name, model)

if __name__ == '__main__':

    main(sys.argv[1])
