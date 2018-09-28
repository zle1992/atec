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
from keras import backend as K
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau


sys.path.append('models')
from CNN import model_conv1D_, ABCNN2, dssm
from RNN import rnn_v1, Siamese_LSTM, my_rnn
from ESIM import esim, decomposable_attention
from ABCNN import ABCNN
from bimpm import bimpm
from MatchZoo import *
sys.path.append('utils/')
sys.path.append('feature/')
import config
from Feats import data_2id, add_hum_feats
from help import *  # score, train_batch_generator, train_batch_generator3,train_batch_generator5,train_test, get_X_Y_from_df
from CutWord import read_cut


def get_model(model_name):
    lr = 0.001
    if model_name == 'bimpm':
        model = bimpm()
    if model_name == 'drmmt':
        model = drmm_tks(num_layer=3, hidden_sizes=[100, 80, 1], topk=20)

    if model_name == 'msrnn':
        model = MATCHSRNN()
    if model_name == 'dssm':
        model = dssm()

    if model_name == 'arc2':
        model = arc2()
    if model_name == 'test':
        model = test()
    if model_name == 'cnn':
        lr = 0.01
        model = model_conv1D_()
    if model_name == 'rnn':

        model = rnn_v1()
    if model_name == 'rnn0':
        model = my_rnn()
    if model_name == 'slstm':

        model = Siamese_LSTM()
    if model_name == 'scnn':

        model = Siamese_CNN()

    if model_name == 'esim':
        lr = 0.01
        model = esim()

    if model_name == 'dam':
        model = decomposable_attention()
    if model_name == 'abcnn':

        model = ABCNN(
            left_seq_len=config.word_maxlen, right_seq_len=config.word_maxlen, depth=2,
            nb_filter=100, filter_widths=[5, 3],
            collect_sentence_representations=False, abcnn_1=True, abcnn_2=True,
            # mode="euclidean",
            # mode="cos",
            mode='dot'
        )

    return model, lr


def load_data():
    path = config.origin_csv
    print('load data')
    data = read_cut(path)  # cut word
    data = data_2id(data)  # 2id
    data = add_hum_feats(data, config.train_featdires)  # 生成特征并加入
    return data


def train_model(x_train, y_train, x_dev, y_dev, model, lr, bst_model_path):

    model_checkpoint = ModelCheckpoint(
        bst_model_path, monitor='val_F1', save_best_only=True, save_weights_only=True, mode='max')
    early_stopping = EarlyStopping(monitor='val_F1', patience=4,
                                   mode='max')
    change_lr = ReduceLROnPlateau(
        monitor='val_F1', mode='max', factor=0.1, epsilon=0.001, min_lr=0.0001, patience=1)

    K.set_value(model.optimizer.lr, lr)
    model.fit(x_train, y_train,
              epochs=10,
              validation_data=(x_dev, y_dev),
              batch_size=config.batch_size,
              class_weight={0: 1, 1: 3},
              callbacks=[model_checkpoint, early_stopping, change_lr,
                         # TensorBoard(log_dir='data/log_dir'),
                         ],
              )

    model.load_weights(bst_model_path)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr), metrics=[Precision, Recall, F1, ])
    return model


def make_train_cv_data(data, model_name, kfolds):


    X_train, Y_train = get_X_Y_from_df(data, config.data_augment, config.shuffer)
    S_train = np.zeros((Y_train.shape[0], 2))

    train_df = pd.DataFrame()
    train_df['pred'] = 0
    train_df['label'] = Y_train
    X, Y = X_train, Y_train
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=kfolds, shuffle=True)
    k = 0

    p, r, f = [], [], []
    for train_index, test_index in kf.split(Y):
        k += 1
        bst_model_path = config.stack_path + \
            "dp_feats_%d_%s_%d.h5" % (len(config.feats), model_name, k)

        model, lr = get_model(model_name)

        x_train = [X[i][train_index, :] for i in range(5)]
        x_dev = [X[i][test_index, :] for i in range(5)]
        y_train = Y[train_index]
        y_dev = Y[test_index]
        id_dev = data.id.values[test_index]
        print('kf: ', k)

        model = train_model(x_train, y_train, x_dev, y_dev,
                            model, lr, bst_model_path)
        pred = model.predict(x_dev, batch_size=config.batch_size)
        pre, rec, f1 = score(y_dev, pred)
        S_train[test_index,0] = id_dev
        S_train[test_index,1] = [i[0] for i in pred]
        p.append(pre)
        r.append(rec)
        f.append(f1)

    train_df['pred'] = S_train[:,1]
    train_df['id'] =  S_train[:,0]
    print('p r f1 ')
    print(np.array([p, r, f, ]).T)
    print('mean :', np.mean(np.array(p)),
          np.mean(np.array(r)), np.mean(np.array(f)))
    train_df.to_csv(config.stack_path + 'train_%s.csv' % (k),
                    index=False, )



    


def do_single_train(model_name, model, lr):
    print('model name', model_name)
    bst_model_path = config.model_dir + \
        "dp_feats_%d_embed_%s.h5" % (len(config.feats), model_name)
    data = load_data()
    train, dev = train_test(data)
    x_train, y_train = get_X_Y_from_df(
        train, config.data_augment, config.shuffer)
    x_dev, y_dev = get_X_Y_from_df(dev, False, False)

    train_model(x_train, y_train, x_dev, y_dev, model, lr, bst_model_path)


def do_train_cv(model_name, model, kfolds, lr):
    data = load_data()
    X_train, Y_train = get_X_Y_from_df(
        data, config.data_augment, config.shuffer)
    make_train_cv_data(X_train, Y_train, model, model_name, kfolds, lr)


def cv(model_name):
    kfolds = 5
    data = load_data()
    make_train_cv_data(data, model_name, kfolds)

def single_train(model_name):
    model, lr = get_model(model_name)
    do_single_train(model_name, model, lr)

if __name__ == '__main__':

    m, model_name = sys.argv[1], sys.argv[2]
    if m == 'cv':
        cv(model_name)
    else:
        single_train(model_name)
