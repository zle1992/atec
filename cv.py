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
from keras.callbacks import TensorBoard,ReduceLROnPlateau
sys.path.append('models')
from CNN import  model_conv1D_,ABCNN2
from RNN import rnn_v1,Siamese_LSTM,my_rnn
from ESIM import esim, decomposable_attention
from ABCNN import ABCNN
from bimpm import bimpm
from MatchZoo import *
sys.path.append('utils/')
sys.path.append('feature/')
import config
from Feats import data_2id,add_hum_feats
from help import score, train_batch_generator5, train_batch_generator3,train_test, get_X_Y_from_df
from CutWord import read_cut,cut_word










def load_data():
    print('load data')  
    in_path = config.origin_csv
    data = read_cut(in_path)
    data = data_2id(data)  # 2id
    data = add_hum_feats(data,config.train_featdires)  # 生成特征并加入

    x_train, y_train = get_X_Y_from_df(data, config.data_augment,config.shuffer)
    print(len(x_train[2]))

    return x_train, y_train


def make_train_cv_data(X_train, Y_train, Model, model_name, epoch_nums, kfolds,lr):

    from keras.models import model_from_json

    json_string = Model.to_json()

    S_train = np.zeros((Y_train.shape[0], epoch_nums))
    S_Y = np.zeros((Y_train.shape[0], 1))

    train_df = pd.DataFrame()
    X, Y = X_train, Y_train
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=kfolds, shuffle=True)
    k = 0
       
    p, r, f = [], [], []    
    for train_index, test_index in kf.split(Y):
        k += 1
        model = model_from_json(json_string, custom_objects=config.CustomObjects)
        model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])


        # x_train = [X[0][train_index, :], X[1][train_index, :], X[2][train_index, :]]
        # x_dev = [X[0][test_index, :], X[1][test_index, :],X[2][test_index, :]]    

        x_train = [X[i][train_index, :] for i in range(5)]#X[1][train_index, :], X[2][train_index, :]]
        x_dev = [X[i][test_index, :] for i in range(5)]       
        y_train=Y[train_index,:]
        y_dev = Y[test_index, :]
        print('kf: ', k)
        for epoch_num in range(epoch_nums): 
            K.set_value(model.optimizer.lr, lr)
            if epoch_num == 5:
                K.set_value(model.optimizer.lr, 0.0001)  
            print('epoch_num: ', epoch_num + 1)

            model.fit_generator(
                train_batch_generator5(x_train, y_train, config.batch_size),
                epochs=1,
                steps_per_epoch=int(y_train.shape[0] / config.batch_size),
                validation_data=(x_dev, y_dev),
                class_weight={0: 1, 1: 3},

            )
            pred = model.predict(x_dev, batch_size=config.batch_size)
            pre, rec, f1 = score(y_dev, pred)
            S_train[test_index, epoch_num] = pred[:, 1]
            print('p r f1 ', pre, rec, f1)
            train_df['epoch_{0}'.format(epoch_num)] = S_train[:, epoch_num]
            train_df['label'] = Y_train[:, 1]
        

        

           
        p.append(pre)
        r.append(rec)
        f.append(f1)
    
        model.save(config.stack_path+"_%s_%s.h5" %
                   (model_name, k))
    print('p r f1 ')
    print(np.array([p, r, f, ]).T)
    print('mean :', np.mean(np.array(p)),
              np.mean(np.array(r)), np.mean(np.array(f)))
    train_df.to_csv(config.stack_path+'train_%s.csv' % (k),
                    index=False, )


def do_train_cv(model_name, model, epoch_nums, kfolds,lr):
    X_train, Y_train = load_data()
    make_train_cv_data(X_train, Y_train, model, model_name, epoch_nums, kfolds,lr)


def main(model_name):
    lr = 0.01
    print('model name', model_name)
    
    if model_name == 'bimpm':
        model = bimpm()
    if model_name == 'my_rnn':
        lr = 0.001
        model = my_rnn()
    if model_name == 'drmmt':
        model = drmm_tks()

    if model_name == 'cnn':
        lr = 0.005
        model = model_conv1D_()
    if model_name == 'slstm':
        lr = 0.001
        model = Siamese_LSTM()

    if model_name == 'esim':
        model = esim()
        lr = 0.001

    if model_name == 'dam':
        model = decomposable_attention()
        lr = 0.001
    if model_name == 'abcnn':

        model = ABCNN(
            left_seq_len=config.word_maxlen, right_seq_len=config.word_maxlen, depth=2,
            nb_filter=100, filter_widths=[5, 3],
            collect_sentence_representations=True, abcnn_1=False, abcnn_2=False,
            #collect_sentence_representations=True, abcnn_1=True, abcnn_2=False,
            # mode="euclidean",
            mode="cos",
            # mode='dot'
            
        )
    do_train_cv(model_name, model, epoch_nums=6, kfolds=5,lr=lr)
    #train(x_train, y_train, x_dev, y_dev, model_name, model)

if __name__ == '__main__':

    main(sys.argv[1])
    # do_cv()
