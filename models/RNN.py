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
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.preprocessing import sequence
from keras.regularizers import l2
from keras import backend as K
from keras.engine.topology import Layer
from keras.backend.tensorflow_backend import set_session
import time
from keras.activations import softmax
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from keras import backend as K
sys.path.append('utils/')
import config

sys.path.append('models/layers/')

from MyPooling import MyMeanPool,MyMaxPool
from MyEmbeding import create_pretrained_embedding
from Cross import cross

def my_rnn():
    emb_layer = create_pretrained_embedding(
        config.word_embed_weight, mask_zero=False)
    lstm_layer = Bidirectional(CuDNNLSTM(250, recurrent_dropout=0.2))

    sequence_1_input = Input(shape=(config.word_maxlen,), dtype="int32")
    embedded_sequences_1 = emb_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(config.word_maxlen,), dtype="int32")
    embedded_sequences_2 = emb_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    magic_input = Input(shape=(len(config.feats),), dtype="float32")
    features_dense = BatchNormalization()(magic_input)
    features_dense = Dense(2, activation="relu")(features_dense)
    features_dense = Dropout(0.2)(features_dense)


    

    addition = add([x1, y1])
    minus_y1 = Lambda(lambda x: -x)(y1)
    merged = add([x1, minus_y1])
    merged = multiply([merged, merged])
    merged = concatenate([merged, addition])
    merged = Dropout(0.4)(merged)

    merged = concatenate([merged, features_dense])
    merged = BatchNormalization()(merged)
    merged = GaussianNoise(0.1)(merged)

    merged = Dense(300, activation="relu")(merged)
    merged = Dropout(0.2)(merged)
    merged = BatchNormalization()(merged)

    out = Dense(2, activation="sigmoid")(merged)

    model = Model(inputs=[sequence_1_input, sequence_2_input, magic_input], outputs=out)
    model.compile(loss="binary_crossentropy",
                  optimizer=Adam(), metrics=['acc'])
    model.summary()
    return model





def rnn_v1(lstm_hidden=50):

    # The embedding layer containing the word vectors
    # Embedding
    emb_layer = create_pretrained_embedding(
        config.word_embed_weight, mask_zero=    True)
    # Define inputs
    seq1 = Input(shape=(config.word_maxlen,))
    seq2 = Input(shape=(config.word_maxlen,))
    magic_input = Input(shape=(len(config.feats),))

    # Run inputs through embedding
    # emb1 = core.Masking(mask_value=0)(emb_layer(seq1))
    # emb2 = core.Masking(mask_value=0)(emb_layer(seq2))
    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)
 
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(2, activation='relu')(magic_dense)

    
    compose = Bidirectional(CuDNNLSTM(lstm_hidden,return_sequences=True))
    compose2 = Bidirectional(CuDNNLSTM(lstm_hidden,return_sequences=True))
  
  
    q1_compare = compose(emb1)
    #q1_compare = BatchNormalization()(q1_compare)
    q1_compare = compose2(q1_compare)
    q1_compare = BatchNormalization()(q1_compare)
    #q1_compare = MyMaxPool(axis=1)(q1_compare)
    q1_compare = GlobalAveragePooling1D()(q1_compare)
    #q1_compare = Dense(256, activation='elu')(q1_compare)

    q2_compare = compose(emb1)
    #q2_compare = BatchNormalization()(q2_compare)
    q2_compare = compose2(q2_compare)
    q2_compare = BatchNormalization()(q2_compare)
    #q2_compare = MyMaxPool(axis=1)(q2_compare)
    q2_compare = GlobalAveragePooling1D()(q2_compare)
    #q2_compare = Dense(256, activation='elu')(q2_compare)
   
    diff = Lambda(lambda x: K.abs(
        x[0] - x[1]), output_shape=(lstm_hidden*2,))([q1_compare, q2_compare])
    mul = Lambda(lambda x: x[0] * x[1],
                 output_shape=(lstm_hidden*2,))([q1_compare, q2_compare])

    x= concatenate([q1_compare,q2_compare,])#diff,mul])

    # x = Dropout(0.5)(x)
    # x = BatchNormalization()(x)
    # x = Dense(256, activation='relu')(x)

    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    pred = Dense(2, activation='softmax')(x)

    model = Model(inputs=[seq1, seq2, magic_input], outputs=pred)

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(), metrics=['acc'])
    model.summary()
    return model





def Siamese_LSTM():

    # The embedding layer containing the word vectors
    # Embedding
    emb_layer = create_pretrained_embedding(
        config.char_embed_weights, mask_zero=False)
    emb_layer_word = create_pretrained_embedding(
        config.word_embed_weights, mask_zero=False)
    # Model variables

    n_hidden = 128

    # Define the shared model
    x = Sequential()
    x.add(emb_layer)
    # # LSTM
    x.add(Bidirectional(CuDNNLSTM(n_hidden,return_sequences=True)))
    x.add(Bidirectional(CuDNNLSTM(n_hidden,return_sequences=True)))
    x.add(BatchNormalization())
    x.add(MyMaxPool(axis=1))
    shared_model = x



    x2 = Sequential()
    x2.add(emb_layer_word)
    # # LSTM
    x2.add(Bidirectional(CuDNNLSTM(n_hidden,return_sequences=True)))
    x2.add(BatchNormalization())
    x2.add(MyMaxPool(axis=1))
    shared_model2 = x2
    # The visible layer
  
    magic_input = Input(shape=(len(config.feats),))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='elu')(magic_dense)


    left_input = Input(shape=(config.word_maxlen,), dtype='int32')
    right_input = Input(shape=(config.word_maxlen,), dtype='int32')
    w1 = Input(shape=(config.word_maxlen,), dtype='int32')
    w2 = Input(shape=(config.word_maxlen,), dtype='int32')

 

    left = shared_model(left_input)
    right = shared_model(right_input)
   
    left_w = shared_model2(w1)
    right_w = shared_model2(w2)

    # Pack it all up into a Manhattan Distance model
    malstm_distance = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True)), output_shape=(
        1,))([left, right])
    #ManDist()([shared_model(left_input), shared_model(right_input)])

             
    cro = cross(left, right,n_hidden*2)
    cro2 = cross(left_w, right_w,n_hidden*2)
    if config.nofeats:
        merge = concatenate([left, right, cro,cro2,malstm_distance])  # , magic_dense, malstm_distance])
    else:   
        merge = concatenate([left, right, cro,cro2, magic_dense, malstm_distance])
    # # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)
    X = concatenate([x,magic_dense])  # , magic_dense, distance_dense])
    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(2, activation='sigmoid')(x)
    model = Model(inputs=[left_input, right_input,w1,w2,
                          magic_input], outputs=[pred])

    model.compile(loss='binary_crossentropy',
                  optimizer="adam", metrics=['acc'])
    model.summary()
    shared_model.summary()
    return model



