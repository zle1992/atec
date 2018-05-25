#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import keras

def train_batch_generator(x_source, y_source, batch):
    q1_source = x_source[0]
    q2_source = x_source[1]
    while True:
        batch_list_x1 = []
        batch_list_x2 = []

        batch_list_y = []
        for q1,q2, y in zip(q1_source,q2_source, y_source):
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


def score(label, pred, gate=0.5):

    if len(label.shape) == 1:
        p = (pred > gate).astype("int")
        p = np.squeeze(p)
        l = label
    else:
        p = np.argmax(pred, axis=1)
        l = np.argmax(label, axis=1)

    #print(confusion_matrix(l, p).view())
    pre_score = precision_score(l, p, pos_label=1, average='binary')
    rec_score = recall_score(l, p, pos_label=1, average='binary')
    f_score = f1_score(l, p)
    return pre_score, rec_score, f_score

def get_X_Y_from_df(data):
    data_q1 = np.array(list(data.q1_cut_id.values))
    data_q2 = np.array(list(data.q2_cut_id.values))

    data_label = data.label.values

    X = [data_q1, data_q2]
    #X = np.concatenate(X,1)
    
    Y = keras.utils.to_categorical(data_label, num_classes=2)
    return X, Y


def train_test(data, test_size=0.1):
    data = data.sample(frac=1, random_state=2018)
    train = data[:int(len(data) * (1 - test_size))]
    test = data[int(len(data) * (1 - test_size)):]

    return train, test