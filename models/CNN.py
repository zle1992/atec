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
def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    pretrained_weights = np.load(pretrained_weights_path)
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[
                          pretrained_weights], trainable=True, **kwargs)
    return embedding

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


def model_conv1D_(lr=0.005):

    # The embedding layer containing the word vectors
    # Embedding
    emb_layer = create_pretrained_embedding(
        config.word_embed_weight, mask_zero=False)
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

    # Define inputs
    seq1 = Input(shape=(config.word_maxlen,))
    seq2 = Input(shape=(config.word_maxlen,))
    magic_input = Input(shape=(len(config.feats),))

    # Run inputs through embedding
    emb1 = emb_layer(seq1)
    emb2 = emb_layer(seq2)

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

    # Add the magic features

    

  

    # # Add the distance features (these are now TFIDF (character and word), Fuzzy matching,
    # # nb char 1 and 2, word mover distance and skew/kurtosis of the sentence
    # # vector)
    # distance_input = Input(shape=(20,))
    # distance_dense = BatchNormalization()(distance_input)
    # distance_dense = Dense(128, activation='relu')(distance_dense)

    # Merge the Magic and distance features with the difference layer

        # , magic_dense, distance_dense])
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)
    merge = concatenate([diff, mul, magic_dense])

    # compose = Bidirectional(LSTM(256))
  
    # q1_compare = compose(emb1)
    # q1_compare = BatchNormalization()(q1_compare)
    # # q1_compare = Dense(256, activation='elu')(q1_compare)

    # q2_compare = compose(emb1)
    # q2_compare = BatchNormalization()(q2_compare)
    # # q2_compare = Dense(256, activation='elu')(q2_compare)
    # print(q2_compare)
    # diff2 = Lambda(lambda x: K.abs(
    #     x[0] - x[1]), output_shape=(512,))([q1_compare, q2_compare])
    # mul2 = Lambda(lambda x: x[0] * x[1],
    #              output_shape=(512,))([q1_compare, q2_compare])

    # merge = concatenate([merge,diff2,mul2])
    # # The MLP that determines the outcome




    # cos = cosine_similarity(mergea, mergeb)
    # print(cos)
    # cos= Flatten()(cos)
    # print(cos)
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[seq1, seq2, magic_input], outputs=pred)
    # model = Model(inputs=[seq1, seq2, magic_input,
    #                       distance_input], outputs=pred)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr), metrics=['acc'])
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

        # # concat attention
        # # (samples, channels, rows, cols)
        # left_embed = merge([left_embed, attention_left],
        #                    mode="concat", concat_axis=1)
        # right_embed = merge([right_embed, attention_right],
        #                     mode="concat", concat_axis=1)

        # # Padding so we have wide convolution
        # left_embed_padded = ZeroPadding2D((filter_width - 1, 0))(left_embed)
        # right_embed_padded = ZeroPadding2D((filter_width - 1, 0))(right_embed)

       
        # left_embed_padded = left_embed
        # right_embed_padded = right_embed

        # # 2D convolutions so we have the ability to treat channels.
        # # Effectively, we are still doing 1-D convolutions.
  
        # my_conv2d = Conv2D(activation="tanh", data_format="channels_first", padding="valid", filters=nb_filter, kernel_size=(filter_width, out_dim))
        # my_conv2d2 = Conv2D(activation="tanh", data_format="channels_first", padding="valid", filters=nb_filter, kernel_size=(filter_width, out_dim))
        # conv_left = my_conv2d(left_embed_padded)
        
        # # Reshape and Permute to get back to 1-D
        # conv_left = (Reshape((conv_left._keras_shape[1], conv_left._keras_shape[2])))(
        #     conv_left)
        # conv_left = Permute((2, 1))(conv_left)


        # conv_right = my_conv2d2(right_embed_padded)

        # # Reshape and Permute to get back to 1-D
        # conv_right = (
        #     Reshape((conv_right._keras_shape[1], conv_right._keras_shape[2])))(conv_right)
        # conv_right = Permute((2, 1))(conv_right)


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

    # Add the magic features
    if config.feats == []:
        magic_input = Input(shape=(1,))
        merge = concatenate([diff, mul])  # , magic_dense, distance_dense])
    else:
        magic_input = Input(shape=(len(config.feats),))
        magic_dense = BatchNormalization()(magic_input)
        magic_dense = Dense(64, activation='relu')(magic_dense)

    # # Add the distance features (these are now TFIDF (character and word), Fuzzy matching,
    # # nb char 1 and 2, word mover distance and skew/kurtosis of the sentence
    # # vector)
    # distance_input = Input(shape=(20,))
    # distance_dense = BatchNormalization()(distance_input)
    # distance_dense = Dense(128, activation='relu')(distance_dense)

    # Merge the Magic and distance features with the difference layer

        # , magic_dense, distance_dense])


    # # The MLP that determines the outcome
    x = Dropout(0.5)(merge)
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


def Siamese_LSTM():

    # The embedding layer containing the word vectors
    # Embedding
    emb_layer = create_pretrained_embedding(
        config.word_embed_weight, mask_zero=False)
    # Model variables

    n_hidden = 50

    # Define the shared model
    x = Sequential()
    x.add(emb_layer)
    # CNN
    # x.add(Conv1D(250, kernel_size=5, activation='relu'))
    # x.add(GlobalMaxPool1D())
    # x.add(Dense(250, activation='relu'))
    # x.add(Dropout(0.3))
    # # x.add(Dense(1, activation='sigmoid'))
    # # LSTM
    x.add(GRU(n_hidden))

    shared_model = x

    # The visible layer
    if config.feats == []:
        magic_input = Input(shape=(1,))
    else:
        magic_input = Input(shape=(len(config.feats),))
    left_input = Input(shape=(config.word_maxlen,), dtype='int32')
    right_input = Input(shape=(config.word_maxlen,), dtype='int32')

    # Pack it all up into a Manhattan Distance model
    malstm_distance = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True)), output_shape=(
        2,))([shared_model(left_input), shared_model(right_input)])
    #ManDist()([shared_model(left_input), shared_model(right_input)])

    left = shared_model(left_input)
    right = shared_model(right_input)
    merge = concatenate([left, right, ])  # , magic_dense, distance_dense])

    # # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(2, activation='sigmoid')(x)
    model = Model(inputs=[left_input, right_input,
                          magic_input], outputs=[pred])

    model.compile(loss='binary_crossentropy',
                  optimizer="adam", metrics=['acc'])
    model.summary()
    shared_model.summary()
    return model
