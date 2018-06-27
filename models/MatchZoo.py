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
from Match      import *
from MyEmbeding import create_pretrained_embedding

def drmm_tks(num_layer=4, hidden_sizes=[256,128,128,64],topk=20):
    emb_layer = create_pretrained_embedding(
        config.word_embed_weight, mask_zero=False)
    q1 = Input(shape=(config.word_maxlen,))
    q2 = Input(shape=(config.word_maxlen,))
    magic_input = Input(shape=(len(config.feats),))

    q1_embed = emb_layer(q1)
    q2_embed = emb_layer(q2)

    mm = Dot(axes=[2, 2], normalize=True)([q1_embed, q2_embed])
    # compute term gating
    w_g = Dense(1)(q1_embed)

    g = Lambda(lambda x: softmax(x, axis=1), output_shape=(
        config.word_maxlen, ))(w_g)
  
    g = Reshape((config.word_maxlen,))(g)
  

    mm_k = Lambda(lambda x: K.tf.nn.top_k(
        x, k=topk, sorted=True)[0])(mm)
  

    for i in range(num_layer):
        mm_k = Dense(hidden_sizes[i], activation='softplus',
                     kernel_initializer='he_uniform', bias_initializer='zeros')(mm_k)
        

    mm_k_dropout = Dropout(rate=0.5)(mm_k)
  

    mm_reshape =  mm_k_dropout #Reshape((config.word_maxlen,))(mm_k_dropout)
   

    mean = Dot(axes=[1, 1])([mm_reshape, g])
  

    out_ = Dense(2, activation='softmax')(mean)


    model = Model(inputs=[q1, q2, magic_input], outputs=out_)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
    model.summary()
    return model

def MATCHSRNN(channel=2):
    emb_layer = create_pretrained_embedding(
        config.word_embed_weight, mask_zero=False)
    q1 = Input(shape=(config.word_maxlen,))
    q2 = Input(shape=(config.word_maxlen,))
    if len(config.feats) == 0:
        magic_input = Input(shape=(1,))
    else:
        magic_input = Input(shape=(len(config.feats),))
    q1_embed = emb_layer(q1)

    q2_embed = emb_layer(q2)

    match_tensor = MatchTensor(channel=channel)([q1_embed, q2_embed])
        
    match_tensor_permute = Permute((2, 3, 1))(match_tensor)
    h_ij = SpatialGRU()(match_tensor)
        
    h_ij_drop = Dropout(rate=0.5)(h_ij)
        

    out_ = Dense(2, activation='softmax')(h_ij_drop)
    
      


    model = Model(inputs=[q1, q2, magic_input], outputs=out_)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def arc2(a1d_kernel_count=256,a1d_kernel_size=3,num_conv2d_layers=1,
        a2d_kernel_counts=[64],
        a2d_kernel_sizes=[[5,5],[5,5]],
        a2d_mpool_sizes=[[2,2],[2,2]]):
    emb_layer = create_pretrained_embedding(
        config.word_embed_weight, mask_zero=False)
    q1 = Input(shape=(config.word_maxlen,))
    q2 = Input(shape=(config.word_maxlen,))
    if len(config.feats) == 0:
        magic_input = Input(shape=(1,))
    else:
        magic_input = Input(shape=(len(config.feats),))
    q1_embed = emb_layer(q1)

    q2_embed = emb_layer(q2)

    q_conv1 = Conv1D(a1d_kernel_count, a1d_kernel_size, padding='same') (q1_embed)

    d_conv1 = Conv1D(a1d_kernel_count, a1d_kernel_size, padding='same') (q2_embed)


    cross = Match(match_type='plus')([q_conv1, d_conv1])


    z = Reshape((config.word_maxlen, config.word_maxlen, -1))(cross)


    for i in range(num_conv2d_layers):
        z = Conv2D(filters=a2d_kernel_counts[i], kernel_size=a2d_kernel_sizes[i], padding='same', activation='relu')(z)
        z = MaxPooling2D(pool_size=(a2d_mpool_sizes[i][0],a2d_mpool_sizes[i][1]))(z)
        

    #dpool = DynamicMaxPooling(self.config['dpool_size'][0], self.config['dpool_size'][1])([conv2d, dpool_index])
    pool1_flat = Flatten()(z)

    pool1_flat_drop = Dropout(rate=0.5)(pool1_flat)


    out_ = Dense(2, activation='softmax')(pool1_flat_drop)

    model = Model(inputs=[q1, q2, magic_input], outputs=out_)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
    model.summary()
    return model


def  test0(alm_kernel_count=64,
    alm_hidden_sizes =[256,512],
        dm_kernel_count=32,
        dm_kernel_size= 3,
        dm_q_hidden_size=32,
        dm_d_mpool=3,
        dm_hidden_sizes=[50],
    ):
    def xor_match(x):
        t1 = x[0]
        t2 = x[1]
        t1_shape = t1.get_shape()
        t2_shape = t2.get_shape()
        t1_expand = K.tf.stack([t1] * t2_shape[1], 2)
        t2_expand = K.tf.stack([t2] * t1_shape[1], 1)
        out_bool = K.tf.equal(t1_expand, t2_expand)
        out = K.tf.cast(out_bool, K.tf.float32)
        return out
    def hadamard_dot(x):
        x1 = x[0]
        x2 = x[1]
        out = x1 * x2
        #out = tf.matmul(x1, x2)
        #out = K.tf.einsum('ij, ijk -> jk', x1, x2)
        return out

    emb_layer = create_pretrained_embedding(
        config.word_embed_weight, mask_zero=False)
    q1 = Input(shape=(config.word_maxlen,))
    q2 = Input(shape=(config.word_maxlen,))
    if len(config.feats) == 0:
        magic_input = Input(shape=(1,))
    else:
        magic_input = Input(shape=(len(config.feats),))
    q1_embed = emb_layer(q1)

    q2_embed = emb_layer(q2)

    lm_xor = Lambda(xor_match)([q1, q2])
 
    #lm_xor_reshape = Reshape((self.config['text1_maxlen'], self.config['text2_maxlen'], 1))(lm_xor)
    #show_layer_info('Reshape', lm_xor_reshape)
    lm_conv = Conv1D(alm_kernel_count,config.word_maxlen, padding='same', activation='tanh')(lm_xor)
  
    lm_conv = Dropout(0.5)(lm_conv)
 
    lm_feat = Reshape((-1,))(lm_conv)

    for hidden_size in alm_hidden_sizes:
        lm_feat = Dense(hidden_size, activation='tanh')(lm_feat)

    lm_drop = Dropout(0.5)(lm_feat)
   
    lm_score = Dense(1)(lm_drop)
    

    dm_q_conv = Conv1D(dm_kernel_count, dm_kernel_size, padding='same', activation='tanh')(q1_embed)

    dm_q_conv = Dropout(0.5)(dm_q_conv)

    dm_q_mp = MaxPooling1D(pool_size = config.word_maxlen)(dm_q_conv)

    dm_q_rep = Reshape((-1,))(dm_q_mp)

    dm_q_rep = Dense(dm_q_hidden_size)(dm_q_rep)
    
    dm_q_rep = Lambda(lambda x: tf.expand_dims(x, 1))(dm_q_rep)

    dm_d_conv1 = Conv1D(dm_kernel_count, dm_kernel_size, padding='same', activation='tanh')(q2_embed)
   
    dm_d_conv1 = Dropout(0.5)(dm_d_conv1)
    
    dm_d_mp = MaxPooling1D(pool_size = dm_d_mpool)(dm_d_conv1)
  
    dm_d_conv2 = Conv1D(dm_kernel_count, 1, padding='same', activation='tanh')(dm_d_mp)
 
    dm_d_conv2 = Dropout(0.5)(dm_d_conv2)

    h_dot = Lambda(hadamard_dot)([dm_q_rep, dm_d_conv2])
  
    dm_feat = Reshape((-1,))(h_dot)

    dm_feat = Dense(hidden_size)(dm_feat)
     
    dm_feat_drop = Dropout(0.5)(dm_feat)
   
    dm_score = Dense(1)(dm_feat_drop)
   
    out_ = Add()([lm_score, dm_score])
    
    
    out_ = Dense(2, activation='softmax')(out_)
    
    model = Model(inputs=[q1, q2, magic_input], outputs=out_)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
    model.summary()
    return model

def test():

    emb_layer = create_pretrained_embedding(
        config.word_embed_weight, mask_zero=False)
    q1 = Input(shape=(config.word_maxlen,))
    q2 = Input(shape=(config.word_maxlen,))
    if len(config.feats) == 0:
        magic_input = Input(shape=(1,))
    else:
        magic_input = Input(shape=(len(config.feats),))
    q1_embed = emb_layer(q1)
    q2_embed = emb_layer(q2)

    

    cross = Dot(axes=[2, 2], normalize=False)([q1_embed, q2_embed])
    
    cross_reshape = Reshape((config.word_maxlen, config.word_maxlen, 1))(cross)

    conv2d = Conv2D(256, 3, padding='same', activation='relu')
    
    conv1 = conv2d(cross_reshape)
    conv1 = MaxPooling2D()(conv1)
    conv1 =  Conv2D(128, 3, padding='same', activation='relu')(conv1)
    pool1 = MaxPooling2D()(conv1)
    pool1_flat = Flatten()(conv1)
    
    pool1_flat_drop = Dropout(rate=0.5)(pool1_flat)
    
    out_ = Dense(128, activation='relu')(pool1_flat_drop)
    out_ = Dense(2, activation='softmax')(out_)
    

    model = Model(inputs=[q1, q2, magic_input], outputs=out_)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['acc'])
    model.summary()
    return model