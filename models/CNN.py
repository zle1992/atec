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
from tensorflow.contrib import learn
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
from keras import backend as K
sys.path.append('utils/')
import config


def create_pretrained_embedding(pretrained_weights_path, trainable=False, **kwargs):
    "Create embedding layer from a pretrained weights array"
    pretrained_weights = np.load(pretrained_weights_path)
    in_dim, out_dim = pretrained_weights.shape
    embedding = Embedding(in_dim, out_dim, weights=[
                          pretrained_weights], trainable=True, **kwargs)
    return embedding


def model_conv1D_():

    # The embedding layer containing the word vectors
    # Embedding
    emb_layer = create_pretrained_embedding(
        config.word_embed_weight, mask_zero=False)

    # 1D convolutions that can iterate over the word vectors
    conv1 = Conv1D(filters=128, kernel_size=1,
                   padding='same', activation='relu')
    conv2 = Conv1D(filters=128, kernel_size=2,
                   padding='same', activation='relu')
    conv3 = Conv1D(filters=128, kernel_size=3,
                   padding='same', activation='relu')
    conv4 = Conv1D(filters=128, kernel_size=4,
                   padding='same', activation='relu')
    conv5 = Conv1D(filters=32, kernel_size=5,
                   padding='same', activation='relu')
    conv6 = Conv1D(filters=32, kernel_size=6,
                   padding='same', activation='relu')

    # Define inputs
    seq1 = Input(shape=(config.word_maxlen,))
    seq2 = Input(shape=(config.word_maxlen,))

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
        x[0] - x[1]), output_shape=(4 * 128 + 2 * 32,))([mergea, mergeb])
    mul = Lambda(lambda x: x[0] * x[1],
                 output_shape=(4 * 128 + 2 * 32,))([mergea, mergeb])

    # Add the magic features
    magic_input = Input(shape=(6,))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)

    # # Add the distance features (these are now TFIDF (character and word), Fuzzy matching,
    # # nb char 1 and 2, word mover distance and skew/kurtosis of the sentence
    # # vector)
    # distance_input = Input(shape=(20,))
    # distance_dense = BatchNormalization()(distance_input)
    # distance_dense = Dense(128, activation='relu')(distance_dense)

    # Merge the Magic and distance features with the difference layer

    merge = concatenate([diff, mul, magic_dense])  # , magic_dense, distance_dense])

    # # The MLP that determines the outcome
    x = Dropout(0.2)(merge)
    x = BatchNormalization()(x)
    x = Dense(300, activation='relu')(x)

    x = Dropout(0.2)(x)
    x = BatchNormalization()(x)
    pred = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=[seq1, seq2,magic_input], outputs=pred)
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
    model = Model(inputs=[q1_input, q2_input], outputs=output)
    model.compile(loss='categorical_crossentropy',
                  optimizer="adam", metrics=['accuracy'])
    model.summary()
    return model


# def get_textcnn3(seq_length, embed_weight, pretrain=False):
#     '''
#     deep cnn conv + maxpooling + conv + maxpooling
#     '''
#     content = Input(shape=(seq_length,), dtype="int32")
#     if pretrain:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#                               'vocab_size'], weights=[embed_weight], output_dim=config['w2v_vec_dim'], trainable=False)
#     else:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#                               'vocab_size'], output_dim=config['w2v_vec_dim'], trainable=True)
#     trans_content = Activation(activation="relu")(
#         BatchNormalization()((TimeDistributed(Dense(256))(embedding(content)))))
#     feat = convs_block2(trans_content)

#     dropfeat = Dropout(0.2)(feat)
#     fc = Activation(activation="relu")(
#         BatchNormalization()(Dense(256)(dropfeat)))
#     output = Dense(2, activation="softmax")(fc)
#     model = Model(inputs=content, outputs=output)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer="adam", metrics=['accuracy'])
#     model.summary()
#     return model


# def get_textcnn2(seq_length, embed_weight, pretrain=False):
#     # 模型结构：词嵌入-卷积池化-卷积池化-flat-drop-softmax

#     main_input = Input(shape=(seq_length,), dtype='float64')

#     # 词嵌入（使用预训练的词向量）

#     if pretrain:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#                               'vocab_size'], weights=[embed_weight], output_dim=config['w2v_vec_dim'], trainable=False)
#     else:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#                               'vocab_size'], output_dim=config['w2v_vec_dim'], trainable=True)
#     embed = embedding(main_input)

#     cnn = Activation(activation='relu')(BatchNormalization()(
#         Convolution1D(filters=256, kernel_size=3, padding='valid')(embed)))
#     cnn = MaxPool1D(pool_size=4)(cnn)

#     cnn = Activation(activation='relu')(BatchNormalization()(
#         Convolution1D(filters=256, kernel_size=3, padding='valid')(cnn)))
#     #cnn = MaxPool1D(pool_size=4)(cnn)
#     cnn = GlobalMaxPool1D()(cnn)
#     #cnn = Flatten()(cnn)
#     drop = Dropout(0.2)(cnn)
#     main_output = Dense(config['number_classes'], activation='softmax')(drop)
#     model = Model(inputs=main_input, outputs=main_output)
#     model.compile(loss='categorical_crossentropy',
#                   optimizer=Adam(), metrics=['accuracy'])
#     return model


# def get_textrnn(seq_length, embed_weight, pretrain=False):
#     # 模型结构：词嵌入-卷积池化*3-拼接-全连接-dropout-全连接

#     main_input = Input(shape=(seq_length,), dtype='float64')

#     # 词嵌入（使用预训练的词向量）

#     if pretrain:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#             'vocab_size'], weights=[embed_weight], output_dim=config['w2v_vec_dim'], trainable=False)
#     else:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#             'vocab_size'], output_dim=config['w2v_vec_dim'], trainable=True)
#     content = embedding(main_input)
#     # trans_content = Activation(activation="relu")(
#     #     BatchNormalization()((TimeDistributed(Dense(256))(embedding(co
#     # print('Build model...')
#     embed = Bidirectional(GRU(256))(content)

#     # merged = layers.add([encoded_sentence, encoded_question])
#     merged = BatchNormalization(embed)
#     merged = Dropout(0.3)(merged)
#     fc = Activation(activation="relu")(
#         BatchNormalization()(Dense(256)(merged)))
#     main_output = Dense(config['number_classes'],
#                         activation='softmax')(fc)

#     model = Model(inputs=main_input, outputs=main_output)
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model


# def get_textrcnn(seq_length, embed_weight, pretrain=False,trainable=False):
#     # 模型结构：词嵌入-卷积池化

#     main_input = Input(shape=(seq_length,), dtype='float64')

#     # 词嵌入（使用预训练的词向量）

#     if pretrain:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#             'vocab_size'], weights=[embed_weight], output_dim=config['w2v_vec_dim'], trainable=trainable)
#     else:
#         embedding = Embedding(name='word_embedding', input_dim=config[
#             'vocab_size'], output_dim=config['w2v_vec_dim'], trainable=True)

#     print('Build model...')
#     content = embedding(main_input)
#     trans_content = Activation(activation="relu")(
#         BatchNormalization()((TimeDistributed(Dense(256))(content))))
#     conv = Activation(activation="relu")(BatchNormalization()(
#         Conv1D(filters=128, kernel_size=5, padding="valid")(trans_content)))

#     cnn1 = conv
#     cnn1 = MaxPool1D(pool_size=5)(cnn1)
#     gru = Bidirectional(GRU(128))(cnn1)

#     merged = Activation(activation="relu")(gru)
#     merged = Dropout(0.2)(merged)
#     main_output = Dense(config['number_classes'],
#                         activation='softmax')(merged)

#     model = Model(inputs=main_input, outputs=main_output)
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     return model


# def model3():
#     sequence_length = x_text.shape[1]  # 56
#     vocabulary_size = config['vocab_size'] + 1  # 18765
#     embedding_dim = 256
#     filter_sizes = [3, 4, 5]
#     num_filters = 512
#     drop = 0.5

#     epochs = 100
#     batch_size = 30

#     # this returns a tensor
#     print("Creating Model...")
#     inputs = Input(shape=(sequence_length,), dtype='int32')
#     embedding = Embedding(input_dim=vocabulary_size + 1, weights=[
#         embed_weight], output_dim=embedding_dim, trainable=True, input_length=config['word_maxlen'])(inputs)
#     # embedding = Embedding(input_dim=vocabulary_size,
#     # output_dim=embedding_dim, input_length=sequence_length)(inputs)

#     print('reshape')
#     reshape = Reshape((sequence_length, embedding_dim, 1))(embedding)

#     conv_0 = Conv2D(num_filters, kernel_size=(filter_sizes[
#         0], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
#     conv_1 = Conv2D(num_filters, kernel_size=(filter_sizes[
#         1], embedding_dim), padding='valid', kernel_initializer='normal', activation='relu')(reshape)
#     conv_2 = Conv2D(num_filters, kernel_size=(filter_sizes[
#         2], embedding_dim), padding='valid', kernel_initializer='normal',
#         activation='relu')(reshape)

#     maxpool_0 = MaxPool2D(pool_size=(
#         sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1), padding='valid')(conv_0)
#     maxpool_1 = MaxPool2D(pool_size=(
#         sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1), padding='valid')(conv_1)
#     maxpool_2 = MaxPool2D(pool_size=(
#         sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1),
#         padding='valid')(conv_2)

#     concatenated_tensor = Concatenate(axis=1)(
#         [maxpool_0, maxpool_1, maxpool_2])
#     flatten = Flatten()(concatenated_tensor)
#     dropout = Dropout(drop)(flatten)
#     output = Dense(units=2, activation='softmax')(dropout)

#     # this creates a model that includes
#     model = Model(inputs=inputs, outputs=output)

#     checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5',
#                                  monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
#     adam = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

#     model.compile(optimizer=adam, loss='binary_crossentropy',
#                   metrics=['accuracy'])
#     print("Traning Model...")
#     model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
#               callbacks=[checkpoint], validation_data=(x_dev, y_dev))  # starts
#     training
