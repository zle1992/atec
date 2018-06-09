#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import keras
import config

def train_batch_generator(x_source, y_source, batch):
    q1_source = x_source[0]
    q2_source = x_source[1]
    while True:
        batch_list_x1 = []
        batch_list_x2 = []

        batch_list_y = []
        for q1, q2, y in zip(q1_source, q2_source, y_source):
            x1 = q1.astype('float32')
            x2 = q2.astype('float32')
            batch_list_x1.append(x1)
            batch_list_x2.append(x2)

            batch_list_y.append(y)
            if len(batch_list_y) == batch:
                yield ([np.array(batch_list_x1), np.array(batch_list_x2)], np.array(batch_list_y))
                batch_list_x1 = []
                batch_list_x2 = []
                batch_list_y = []


def train_batch_generator3(x_source, y_source, batch):
    q1_source = x_source[0]
    q2_source = x_source[1]
    f1_source = x_source[2]
    while True:
        batch_list_x1 = []
        batch_list_x2 = []
        batch_list_x3 = []

        batch_list_y = []
        for q1, q2, f1, y in zip(q1_source, q2_source,f1_source, y_source):
            x1 = q1.astype('float32')
            x2 = q2.astype('float32')
            x3 = f1.astype('float32')
            batch_list_x1.append(x1)
            batch_list_x2.append(x2)
            batch_list_x3.append(f1)

            batch_list_y.append(y)
            if len(batch_list_y) == batch:
                yield ([np.array(batch_list_x1), np.array(batch_list_x2), np.array(batch_list_x3)],np.array(batch_list_y))
                batch_list_x1 = []
                batch_list_x2 = []
                batch_list_x3 = []
                batch_list_y = []


def score(label, pred):

    if len(label.shape) == 1:
        p = pred
        l = label
    else:
        p = np.argmax(pred, axis=1)
        l = np.argmax(label, axis=1)

    #print(confusion_matrix(l, p).view())
    pre_score = precision_score(l, p, pos_label=1, average='binary')
    rec_score = recall_score(l, p, pos_label=1, average='binary')
    f_score = f1_score(l, p)
    return pre_score, rec_score, f_score


def get_X_Y_from_df(data, data_augment=True,sampling = True):
    sampling = True
    if sampling:
        data1 = data[data.label==1]
        data2 = data1.append(data1).append(data1).append(data1)
        data = data.append(data2)


    data_q1 = np.array(list(data.q1_cut_id.values))
    data_q2 = np.array(list(data.q2_cut_id.values))

    magic_feat = np.array(list(data.magic_feat.values))
    data_label = data.label.values

    if data_augment:
        data_label = np.concatenate([data_label, data_label], 0)
        X = [np.concatenate([data_q1, data_q2], 0),
             np.concatenate([data_q2, data_q1], 0),
             np.concatenate([magic_feat, magic_feat], 0)
             ]

    else:
        data_label = data_label
        X = [data_q1, data_q2, magic_feat]
    Y = keras.utils.to_categorical(data_label, num_classes=2)
    return X, Y


def train_test(data, test_size=0.1):
    data = data.sample(frac=1, random_state=2018)
    train = data[:int(len(data) * (1 - test_size))]
    test = data[int(len(data) * (1 - test_size)):]

    return train, test
