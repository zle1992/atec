#!/usr/bin/python
# -*- coding: utf-8 -*-
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import keras
import config
np.random.seed(seed=111)

from keras.callbacks import Callback
from keras import backend as K

def Recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall
def Precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def F1(y_true, y_pred):
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1



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
        for q1, q2, f1, y in zip(q1_source, q2_source, f1_source, y_source):
            x1 = q1.astype('float32')
            x2 = q2.astype('float32')
            x3 = f1.astype('float32')
            batch_list_x1.append(x1)
            batch_list_x2.append(x2)
            batch_list_x3.append(f1)

            batch_list_y.append(y)
            if len(batch_list_y) == batch:
                yield ([np.array(batch_list_x1), np.array(batch_list_x2), np.array(batch_list_x3)], np.array(batch_list_y))
                batch_list_x1 = []
                batch_list_x2 = []
                batch_list_x3 = []
                batch_list_y = []


def train_batch_generator5(x_source, y_source, batch):
    q1_source = x_source[0]
    q2_source = x_source[1]
    q3_source = x_source[2]
    q4_source = x_source[3]
    q5_source = x_source[4]
    while True:
        batch_list_x1 = []
        batch_list_x2 = []
        batch_list_x3 = []
        batch_list_x4 = []
        batch_list_x5 = []

        batch_list_y = []
        for q1, q2, q3, q4, q5, y in zip(q1_source, q2_source, q3_source, q4_source, q5_source, y_source):
            x1 = q1.astype('float32')
            x2 = q2.astype('float32')
            x3 = q3.astype('float32')
            x4 = q4.astype('float32')
            x5 = q5.astype('float32')

            batch_list_x1.append(x1)
            batch_list_x2.append(x2)
            batch_list_x3.append(x3)
            batch_list_x4.append(x4)
            batch_list_x5.append(x5)

            batch_list_y.append(y)
            if len(batch_list_y) == batch:
                yield ([np.array(batch_list_x1), np.array(batch_list_x2),  np.array(batch_list_x3), np.array(batch_list_x4),
                    np.array(batch_list_x5)], np.array(batch_list_y))
                batch_list_x1 = []
                batch_list_x2 = []
                batch_list_x3 = []
                batch_list_x4 = []
                batch_list_x5 = []
                batch_list_y = []


def score(label, pred):
    
    if len(pred[0])==1:
        pred = [int(x>0.5) for x in pred]
        p=pred
        l=label
    else:
        p=np.argmax(pred, axis=1)
        l=np.argmax(label, axis=1)

    # print(confusion_matrix(l, p).view())
    pre_score=precision_score(l, p, pos_label=1, average='binary')
    rec_score=recall_score(l, p, pos_label=1, average='binary')
    f_score=f1_score(l, p)
    return pre_score, rec_score, f_score


def get_X_Y_from_df(data, data_augment=True, shuffer=True):
    #data=data[:200]
    data['q1_cut_id']=data['q1_cut_id'].map(lambda x: x[:config.char_maxlen])
    data['q2_cut_id']=data['q2_cut_id'].map(lambda x: x[:config.char_maxlen])
    data['q1_word_id']=data['q1_word_id'].map(lambda x: x[:config.word_maxlen])
    data['q2_word_id']=data['q2_word_id'].map(lambda x: x[:config.word_maxlen])


    #data=data[:110]
    sampling=False
    if sampling:
        data1=data[data.label == 1]
        data2=data1.append(data1).append(data1).append(data1)
        data=data.append(data2)
    def ss(a):
        lens=len(a)
        a1=[i for i in a if i != 0]
        np.random.shuffle(a1)
        res=a1 + [0] * (lens - len(a1))
        return res
    if shuffer:
        data1=data.copy()
        data1['q1_cut_id']=data1['q1_cut_id'].map(lambda x: ss(x))
        data1['q2_cut_id']=data1['q2_cut_id'].map(lambda x: ss(x))
        data=data.append(data1)

    data_q1=np.array(list(data.q1_cut_id.values))
    data_q2=np.array(list(data.q2_cut_id.values))
    data_q1_word=np.array(list(data['q1_word_id'].values))
    data_q2_word=np.array(list(data['q2_word_id'].values))
    magic_feat=np.array(list(data.magic_feat.values))

    data_label=data.label.values

    if data_augment:
        data_label=np.concatenate([data_label, data_label], 0)
        X=[np.concatenate([data_q1, data_q2], 0),
             np.concatenate([data_q2, data_q1], 0),
             np.concatenate([magic_feat, magic_feat], 0)
             ]

    else:
        data_label=data_label
        X=[data_q1, data_q2, data_q2_word, data_q2_word, magic_feat]
    Y=data_label
    # keras.utils.to_categorical(data_label, num_classes=2)

    print('magic_feat', magic_feat.shape)

    return X, Y


def train_test(data, test_size=0.1):
    data=data.sample(frac=1, random_state=2017)
    train=data[:int(len(data) * (1 - test_size))]
    test=data[int(len(data) * (1 - test_size)):]

    return train, test
