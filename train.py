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

sys.path.append('models')
from CNN import cnn_v1, cnn_v2, model_conv1D_
from ESIM import esim, decomposable_attention
from ABCNN import ABCNN
sys.path.append('utils/')
import config

from process import read_hdf, make_w2v,load_pre_train_w2v
from help import score, train_batch_generator, train_batch_generator3,train_test, get_X_Y_from_df

def load_data():
    path = config.origin_csv
    print('load data')
    data = read_hdf(path)
    train, dev = train_test(data)
    x_train, y_train = get_X_Y_from_df(train, config.data_augment)
    x_dev, y_dev = get_X_Y_from_df(dev, False)
    print('train shape', x_train[0].shape)
    print('dev shape', x_dev[0].shape)
    return x_train, y_train,x_dev, y_dev

def train(x_train, y_train,x_dev, y_dev,model_name, model):
    for i in range(15):
        if i == 9:
            K.set_value(model.optimizer.lr, 0.0001)

        model.fit_generator(
            train_batch_generator3(x_train, y_train, config.batch_size),
            epochs=1,
            steps_per_epoch=int(y_train.shape[0] / config.batch_size),
            validation_data=(x_dev, y_dev),
            class_weight={0: 1, 1: 3},

        )
        pred = model.predict(x_dev, batch_size=config.batch_size)
        pre, rec, f1 = score(y_dev, pred)
        model.save(config.model_dir + "/dp_embed_%s_%s.h5" %
                   (model_name, f1))
        print('p r f1 ', pre, rec, f1)


def main(model_name):
    print('model name', model_name)
    path = config.origin_csv
    x_train, y_train,x_dev, y_dev = load_data()
    
    # if model_name == 'cnn1':
    #     model = cnn_v1(config.word_maxlen,
    #                    embed_weights, pretrain=True)
    # if model_name == 'cnn2':
    #     model = cnn_v2(config.word_maxlen,
    #                    embed_weights, pretrain=True)

    if model_name == 'cnn':

        model = model_conv1D_()

    if model_name == 'esim':
        model = esim()

    if model_name == 'dam':
        model = decomposable_attention()
    if model_name == 'abcnn':

        model = ABCNN(
            left_seq_len=config.word_maxlen, right_seq_len=config.word_maxlen, depth=2,
            nb_filter=300, filter_widths=[4, 3],
            collect_sentence_representations=True, abcnn_1=True, abcnn_2=True,
            mode="euclidean",
            # mode="cos"
        )

    train(x_train, y_train,x_dev, y_dev,model_name, model)

if __name__ == '__main__':

    main(sys.argv[1])
