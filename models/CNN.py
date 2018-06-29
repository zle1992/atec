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
from MatchTensor import *
from SpatialGRU import *
from MyEmbeding import create_pretrained_embedding
from Cross import cross
from Attention import Attention
from help import *

def cosine_similarity( x1, x2):
        """Compute cosine similarity.
        # Arguments:
            x1: (..., embedding_size)
            x2: (..., embedding_size)
        """
        cos = K.sum(x1 * x2, axis=-1)
        x1_norm = K.sqrt(K.maximum(K.sum(K.square(x1), axis=-1), 1e-6))
        x2_norm = K.sqrt(K.maximum(K.sum(K.square(x2), axis=-1),  1e-6))
        cos = cos / x1_norm / x2_norm
        return cos


def cnn_help(emb1,emb2):
    
    nbfilters=[256,246,256,128]#,64,32]
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=nbfilters[0], kernel_size=1,
                   padding='same', activation='relu')
    conv2 = Conv1D(filters=nbfilters[1], kernel_size=2,
                   padding='same', activation='relu')
    conv3 = Conv1D(filters=nbfilters[2], kernel_size=3,
                   padding='same', activation='relu')
    conv4 = Conv1D(filters=nbfilters[3], kernel_size=4,
                   padding='same', activation='relu')
    # conv5 = Conv1D(filters=nbfilters[4], kernel_size=5,
    #                padding='same', activation='relu')
    # conv6 = Conv1D(filters=nbfilters[5], kernel_size=6,
    #                padding='same', activation='relu')

   

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv1b = conv1(emb2)
    glob1b = GlobalAveragePooling1D()(conv1b)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)
    conv2b = conv2(emb2)
    glob2b = GlobalAveragePooling1D()(conv2b)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)
    conv3b = conv3(emb2)
    glob3b = GlobalAveragePooling1D()(conv3b)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)
    conv4b = conv4(emb2)
    glob4b = GlobalAveragePooling1D()(conv4b)

    # conv5a = conv5(emb1)
    # glob5a = GlobalAveragePooling1D()(conv5a)
    # conv5b = conv5(emb2)
    # glob5b = GlobalAveragePooling1D()(conv5b)

    # conv6a = conv6(emb1)
    # glob6a = GlobalAveragePooling1D()(conv6a)
    # conv6b = conv6(emb2)
    # glob6b = GlobalAveragePooling1D()(conv6b)

    mergea = concatenate([glob1a, glob2a, glob3a, glob4a,])# glob5a,glob6a])
    mergeb = concatenate([glob1b, glob2b, glob3b, glob4b,])# glob5b,glob6b])

    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different
    # measure of equalness
    diff = Lambda(lambda x: K.abs(
        x[0] - x[1]), output_shape=(sum(nbfilters),))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1],
                 output_shape=(sum(nbfilters),))([mergea, mergeb])
    add  = Lambda(lambda x: x[0] + x[1],
                 output_shape=(sum(nbfilters),))([mergea, mergeb])

 
    # merge = concatenate([mergea,mergeb,diff, mul,add])
    cro=cross(mergea,mergeb,sum(nbfilters))
    merge = concatenate([mergea,mergeb,cro])
    return merge



def cnn_help2(emb1,emb2):
    
    nbfilters=[256]
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=nbfilters[0], kernel_size=2,
                   padding='same', activation='relu')
    
    

    # Run through CONV + GAP layers
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv1b = conv1(emb2)
    glob1b = GlobalAveragePooling1D()(conv1b)

    
    mergea =glob1a 
    mergeb = glob1b
    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different
    # measure of equalness
    diff = Lambda(lambda x: K.abs(
        x[0] - x[1]), output_shape=(sum(nbfilters),))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1],
                 output_shape=(sum(nbfilters),))([mergea, mergeb])
    add  = Lambda(lambda x: x[0] + x[1],
                 output_shape=(sum(nbfilters),))([mergea, mergeb])

    cro=cross(mergea,mergeb,sum(nbfilters))
    merge = concatenate([mergea,mergeb,cro])
    return cro
def model_conv1D_(lr=0.005):

 
    # Embedding
    emb_layer = create_pretrained_embedding(
        config.char_embed_weights, mask_zero=False)

    emb_layer_word = create_pretrained_embedding(
        config.word_embed_weights, mask_zero=False)
   
    seq1_char = Input(shape=(config.word_maxlen,),name='q1_c')
    seq2_char = Input(shape=(config.word_maxlen,),name='q2_c')

    seq1_word = Input(shape=(config.word_maxlen,),name='q1_w')
    seq2_word = Input(shape=(config.word_maxlen,),name='q2_w')
    magic_input = Input(shape=(len(config.feats),))

    
    emb1_char = emb_layer(seq1_char)
    emb2_char = emb_layer(seq2_char)
    
    emb1_word = emb_layer_word(seq1_word)
    emb2_word = emb_layer_word(seq2_word)


    
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)

    match_list_char = cnn_help(emb1_char,emb2_char)
    match_list_word = cnn_help2(emb1_word,emb2_word)
    merge = concatenate([match_list_char,match_list_word,magic_dense])

    # x = Dropout(0.5)(merge)
    # x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(merge)


    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    pred = Dense(1, activation='sigmoid')(x)
    #model = Model(inputs=[seq1_char, seq2_char, magic_input], outputs=pred)
    model = Model(inputs=[seq1_char, seq2_char,seq1_word,seq2_word, magic_input], outputs=pred)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr),metrics = [Precision,Recall,F1,])
    model.summary()
    return model





def dssm(lstmsize=20):
    # Embedding
    emb_layer_char = create_pretrained_embedding(
        config.char_embed_weights, trainable=True,mask_zero=False)

    emb_layer_word = create_pretrained_embedding(
        config.word_embed_weights, trainable=False,mask_zero=False)

    char_weights = np.load(config.char_embed_weights)
    word_weights = np.load(config.word_embed_weights)

    input1 = Input(shape=(config.word_maxlen,))
    input2 = Input(shape=(config.word_maxlen,))
    input3 = Input(shape=(len(config.feats),))
    embed1 =emb_layer_word# Embedding(word_weights.shape)
    lstm0 = CuDNNLSTM(lstmsize,return_sequences = True)
    lstm1 = Bidirectional(CuDNNLSTM(lstmsize))
    lstm2 = CuDNNLSTM(lstmsize)
    att1 = Attention(config.word_maxlen)
    den = Dense(64,activation = 'tanh')

    # att1 = Lambda(lambda x: K.max(x,axis = 1))
    v3 = embed1(input3)
    v1 = embed1(input1)
    v2 = embed1(input2)
    v11 = lstm1(v1)
    v22 = lstm1(v2)
    v1ls = lstm2(lstm0(v1))
    v2ls = lstm2(lstm0(v2))
    v1 = Concatenate(axis=1)([att1(v1),v11])
    v2 = Concatenate(axis=1)([att1(v2),v22])

    input1c = Input(shape=(config.word_maxlen,))
    input2c = Input(shape=(config.word_maxlen,))
    embed1c = emb_layer_char#Embedding(char_weights.shape)
    lstm1c = Bidirectional(CuDNNLSTM(56,return_sequences = True))
    lstm2c = Bidirectional(CuDNNLSTM(56))
    att1c = Attention(config.word_maxlen)
    v1c = embed1(input1c)
    v2c = embed1(input2c)
    v11c = lstm1c(v1c)
    v22c = lstm1c(v2c)
    v11c = lstm2c(v11c)
    v22c = lstm2c(v22c)
    v1c = Concatenate(axis=1)([att1c(v1c),v11c])
    v2c = Concatenate(axis=1)([att1c(v2c),v22c])


    mul = Multiply()([v1,v2])
    sub = Lambda(lambda x: K.abs(x))(Subtract()([v1,v2]))
    maximum = Maximum()([Multiply()([v1,v1]),Multiply()([v2,v2])])
    mulc = Multiply()([v1c,v2c])
    subc = Lambda(lambda x: K.abs(x))(Subtract()([v1c,v2c]))
    maximumc = Maximum()([Multiply()([v1c,v1c]),Multiply()([v2c,v2c])])
    sub2 = Lambda(lambda x: K.abs(x))(Subtract()([v1ls,v2ls]))
    

    matchlist = Concatenate(axis=1)([mul,sub,mulc,subc,maximum,maximumc,sub2])
    matchlist = Dropout(0.05)(matchlist)

    matchlist = Concatenate(axis=1)([Dense(32,activation = 'relu')(matchlist),Dense(48,activation = 'sigmoid')(matchlist)])
    res = Dense(2, activation = 'sigmoid')(matchlist)


    model = Model(inputs=[input1c, input2c,input1, input2,  input3], outputs=res)
    model.compile(optimizer=Adam(lr = 0.001), loss="binary_crossentropy", metrics=['acc'])
    model.summary()
    return model







def compute_cos_match_score(l_r):
    l, r = l_r
    return K.batch_dot(
        K.l2_normalize(l, axis=-1),
        K.l2_normalize(r, axis=-1),
        axes=[2, 2]
    )


def compute_euclidean_match_score(l_r):
    l, r = l_r
    denominator = 1. + K.sqrt(
        -2 * K.batch_dot(l, r, axes=[2, 2]) +
        K.expand_dims(K.sum(K.square(l), axis=2), 2) +
        K.expand_dims(K.sum(K.square(r), axis=2), 1)
    )
    denominator = K.maximum(denominator, K.epsilon())
    return 1. / denominator



def MatchScore(l, r, mode="euclidean"):
    if mode == "euclidean":
        return merge(
            [l, r],
            mode=compute_euclidean_match_score,
            output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1])
        )
    elif mode == "cos":
        return merge(
            [l, r],
            mode=compute_cos_match_score,
            output_shape=lambda shapes: (None, shapes[0][1], shapes[1][1])
        )
    elif mode == "dot":
        return merge([l, r], mode="dot")
    else:
        raise ValueError("Unknown match score mode %s" % mode)

def convs_block(data, convs=[3, 4, 5], f=256):
    pools = []
    for c in convs:
        conv = Activation(activation="relu")(BatchNormalization()(
            Conv1D(filters=f, kernel_size=c, padding="valid")(data)))
        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools)
def ABCNN2(
    left_seq_len, right_seq_len, nb_filter, filter_widths,
    depth=2, dropout=0.5, abcnn_1=True, abcnn_2=True, collect_sentence_representations=False, mode="euclidean", batch_normalize=True
):
    assert depth >= 1, "Need at least one layer to build ABCNN"
    assert not (
        depth == 1 and abcnn_2), "Cannot build ABCNN-2 with only one layer!"
    if type(filter_widths) == int:
        filter_widths = [filter_widths] * depth
    assert len(filter_widths) == depth

    print("Using %s match score" % mode)

    left_sentence_representations = []
    right_sentence_representations = []
    if len(config.feats)==0:
        magic_input = Input(shape=(1,))
    else:
        magic_input = Input(shape=(len(config.feats),))


    left_input = Input(shape=(left_seq_len, ))
    right_input = Input(shape=(right_seq_len,))

    # Embedding
    pretrained_weights = np.load(config.word_embed_weight)
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[
                          pretrained_weights], trainable=True,)
    left_embed = embedding(left_input)
    right_embed = embedding(right_input)


    left_embed = BatchNormalization()(left_embed)
    right_embed = BatchNormalization()(right_embed)

    filter_width = filter_widths.pop(0)
    if abcnn_1:
        match_score = MatchScore(left_embed, right_embed, mode=mode)

        # compute attention
        attention_left = TimeDistributed(
            Dense(out_dim, activation="relu"), input_shape=(left_seq_len, right_seq_len))(match_score)
        match_score_t = Permute((2, 1))(match_score)
        attention_right = TimeDistributed(
            Dense(out_dim, activation="relu"), input_shape=(right_seq_len, left_seq_len))(match_score_t)

        left_reshape = Reshape((1, attention_left._keras_shape[
                               1], attention_left._keras_shape[2]))
        right_reshape = Reshape((1, attention_right._keras_shape[
                                1], attention_right._keras_shape[2]))

        attention_left = left_reshape(attention_left)
        left_embed = left_reshape(left_embed)

        attention_right = right_reshape(attention_right)
        right_embed = right_reshape(right_embed)

   

    attention_left = (Reshape((attention_left._keras_shape[2], attention_left._keras_shape[3])))(attention_left)
    

    attention_right = (Reshape((attention_right._keras_shape[2], attention_right._keras_shape[3])))(attention_right)
    
    print(attention_left)
    
    conv_left = Dropout(dropout)(attention_left)
    conv_right = Dropout(dropout)(attention_right)
    nbfilters = [128, 128, 128, 128, 32, 32]

    # nbfilters=[512,512,256,128,64,32]
    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=nbfilters[0], kernel_size=1,
                   padding='same', activation='relu')
    conv2 = Conv1D(filters=nbfilters[1], kernel_size=2,
                   padding='same', activation='relu')
    conv3 = Conv1D(filters=nbfilters[2], kernel_size=3,
                   padding='same', activation='relu')
    conv4 = Conv1D(filters=nbfilters[3], kernel_size=4,
                   padding='same', activation='relu')
    conv5 = Conv1D(filters=nbfilters[4], kernel_size=5,
                   padding='same', activation='relu')
    conv6 = Conv1D(filters=nbfilters[5], kernel_size=6,
                   padding='same', activation='relu')

    # Run through CONV + GAP layers
    emb1 = conv_left
    emb2 = conv_right
    conv1a = conv1(emb1)
    glob1a = GlobalAveragePooling1D()(conv1a)
    conv1b = conv1(conv_right)
    glob1b = GlobalAveragePooling1D()(conv1b)

    conv2a = conv2(emb1)
    glob2a = GlobalAveragePooling1D()(conv2a)
    conv2b = conv2(emb2)
    glob2b = GlobalAveragePooling1D()(conv2b)

    conv3a = conv3(emb1)
    glob3a = GlobalAveragePooling1D()(conv3a)
    conv3b = conv3(emb2)
    glob3b = GlobalAveragePooling1D()(conv3b)

    conv4a = conv4(emb1)
    glob4a = GlobalAveragePooling1D()(conv4a)
    conv4b = conv4(emb2)
    glob4b = GlobalAveragePooling1D()(conv4b)

    conv5a = conv5(emb1)
    glob5a = GlobalAveragePooling1D()(conv5a)
    conv5b = conv5(emb2)
    glob5b = GlobalAveragePooling1D()(conv5b)

    conv6a = conv6(emb1)
    glob6a = GlobalAveragePooling1D()(conv6a)
    conv6b = conv6(emb2)
    glob6b = GlobalAveragePooling1D()(conv6b)

    mergea = concatenate([glob1a, glob2a, glob3a, glob4a, glob5a, glob6a])
    mergeb = concatenate([glob1b, glob2b, glob3b, glob4b, glob5b, glob6b])

    # We take the explicit absolute difference between the two sentences
    # Furthermore we take the multiply different entries to get a different
    # measure of equalness
    diff = Lambda(lambda x: K.abs(
        x[0] - x[1]), output_shape=(sum(nbfilters),))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1],
                 output_shape=(sum(nbfilters),))([mergea, mergeb])





    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)
    x = concatenate([diff, mul, magic_dense])

    # # The MLP that determines the outcome
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    pred = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[left_input, right_input, magic_input], outputs=pred)
    # model = Model(inputs=[seq1, seq2, magic_input,
    #                       distance_input], outputs=pred)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
    model.summary()
    return model

def convs_block(data, convs=[3, 3, 4, 5, 5, 7, 7], f=256):
    pools = []
    for c in convs:
        conv = Activation(activation="relu")(BatchNormalization()(
            Conv1D(filters=f, kernel_size=c, padding="valid")(data)))
        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools)


def convs_block2(data, convs=[3, 4, 5], f=256, name="conv_feat"):
    pools = []
    for c in convs:
        conv = Activation(activation="relu")(BatchNormalization()(
            Conv1D(filters=f, kernel_size=c, padding="valid")(data)))
        conv = MaxPool1D(pool_size=10)(conv)
        conv = Activation(activation="relu")(BatchNormalization()(
            Conv1D(filters=f, kernel_size=c, padding="valid")(conv)))

        pool = GlobalMaxPool1D()(conv)
        pools.append(pool)
    return concatenate(pools, name=name)


def cnn_v2(seq_length, embed_weight, pretrain=False):

    q1_input = Input(shape=(seq_length,), dtype="int32")
    q2_input = Input(shape=(seq_length,), dtype="int32")
    in_dim, out_dim = embed_weight.shape
    embedding = Embedding(input_dim=in_dim, weights=[
        embed_weight], output_dim=out_dim, trainable=False)

    q1 = Activation(activation="relu")(
        BatchNormalization()((TimeDistributed(Dense(256))(embedding(q1_input)))))
    q2 = Activation(activation="relu")(
        BatchNormalization()((TimeDistributed(Dense(256))(embedding(q2_input)))))

    q1_feat = convs_block(q1,)
    q2_feat = convs_block(q2,)

    q1_feat = Dropout(0.5)(q1_feat)
    q2_feat = Dropout(0.5)(q2_feat)

    q1_q2 = concatenate([q1_feat, q2_feat])

    fc = Activation(activation="relu")(
        BatchNormalization()(Dense(256)(q1_q2)))
    output = Dense(2, activation="softmax")(fc)
    print(output)
    model = Model(inputs=[q1_input, q2_input], outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model


def cnn_v1(seq_length, embed_weight, pretrain=False):

    q1_input = Input(shape=(seq_length,), dtype="int32")
    q2_input = Input(shape=(seq_length,), dtype="int32")

    in_dim, out_dim = embed_weight.shape
    embedding = Embedding(input_dim=in_dim, weights=[
        embed_weight], output_dim=out_dim, trainable=False)

    q1_q2 = concatenate([q1_input, q2_input])
    q1_q2 = Activation(activation="relu")(
        BatchNormalization()((TimeDistributed(Dense(256))(embedding(q1_q2)))))

    q1_q2 = convs_block(q1_q2)

    q1_q2 = Dropout(0.5)(q1_q2)
    fc = Activation(activation="relu")(
        BatchNormalization()(Dense(256)(q1_q2)))
    output = Dense(2, activation="softmax")(fc)
    print(output)
    model = Model(inputs=[q1_input, q2_input, magic_input], outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model
