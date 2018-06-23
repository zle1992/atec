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
from keras.activations import softmax
from keras import backend
# Model Load
sys.path.append('utils/')
sys.path.append('feature/')
import config
from Feats import data_2id, add_hum_feats
from CutWord import cut_word
from help import get_X_Y_from_df
import pandas as pd

def load_data(in_path):
    print('load data')
    data = cut_word(in_path, config.cut_char_level)
    data = data_2id(data)  # 2id
    data = add_hum_feats(data, '')  # 生成特征并加入

    return data


def main(in_path, out_path):

    for file in os.listdir('./'):
        if file.endswith('.h5'):
            model_path = file
    data = load_data(in_path)
    X, _ = get_X_Y_from_df(data, False, False)
    print('load model and predict')
    model = load_model(model_path, custom_objects={"softmax": softmax})
    test_pred = model.predict(X, batch_size=config.batch_size)
    print('save submit file')
    data['label'] = [int(x > 0.5) for x in test_pred[:, 1]]
    data[['id', 'label']].to_csv(out_path, index=False, header=None, sep='\t')



if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])

