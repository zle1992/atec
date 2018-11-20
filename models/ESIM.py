import numpy as np
import pandas as pd
from keras.layers import *

from keras.activations import softmax
from keras.models import Model
from keras.models import Sequential
from keras.optimizers import Nadam, Adam
from keras.regularizers import l2
import keras.backend as K
import sys
sys.path.append('utils/')
import config
sys.path.append('models/layers/')
from help import *


from MyPooling import MyMeanPool,MyMaxPool
from MyEmbeding import  create_pretrained_embedding
from Cross import cross,distence
MAX_LEN = config.word_maxlen 


class Attention(Layer):
    def __init__(self, step_dim=config.word_maxlen,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
        self.supports_masking = True
        # self.init = initializations.get('glorot_uniform')
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        # eij = K.dot(x, self.W) TF backend doesn't support it

        # features_dim = self.W.shape[0]
        # step_dim = x._keras_shape[1]

        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)), K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        # return input_shape[0], input_shape[-1]
        return input_shape[0], self.features_dim

def unchanged_shape(input_shape):
    "Function for Lambda layer"
    return input_shape


def substract(input_1, input_2):
    "Substract element-wise"
    neg_input_2 = Lambda(lambda x: -x, output_shape=unchanged_shape)(input_2)
    out_ = Add()([input_1, neg_input_2])
    return out_


def submult(input_1, input_2):
    "Get multiplication and subtraction then concatenate results"
    mult = Multiply()([input_1, input_2])
    sub = substract(input_1, input_2)
    out_ = Concatenate()([sub, mult])
    return out_





def apply_multiple(input_, layers):
    "Apply layers to input then concatenate result"
    if not len(layers) > 1:
        raise ValueError('Layers list should contain more than 1 layer')
    else:
        agg_ = []
        for layer in layers:
            agg_.append(layer(input_))
        out_ = Concatenate()(agg_)
    return out_


def time_distributed(input_, layers):
    "Apply a list of layers in TimeDistributed mode"
    out_ = []
    node_ = input_
    for layer_ in layers:
        node_ = TimeDistributed(layer_)(node_)
    out_ = node_
    return out_


def soft_attention_alignment(input_1, input_2):
    "Align text representation with neural soft attention"
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1),
                     output_shape=unchanged_shape)(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2),
                                     output_shape=unchanged_shape)(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])
    return in1_aligned, in2_aligned


def decomposable_attention(pretrained_embedding=config.word_embed_weights,
                           projection_dim=300, projection_hidden=0, projection_dropout=0.2,
                           compare_dim=500, compare_dropout=0.2,
                           dense_dim=300, dense_dropout=0.2,
                           lr=1e-3, activation='elu', maxlen=MAX_LEN):
    # Based on: https://arxiv.org/abs/1606.01933

    
    magic_input = Input(shape=(len(config.feats),))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)

    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))
    q1_w = Input(name='q1_w', shape=(maxlen,))
    q2_w = Input(name='q2_w', shape=(maxlen,))
    # Embedding
    embedding = create_pretrained_embedding(pretrained_embedding,
                                            mask_zero=False)
    q1_embed = embedding(q1)
    q2_embed = embedding(q2)

    # Projection
    projection_layers = []
    if projection_hidden > 0:
        projection_layers.extend([
            Dense(projection_hidden, activation=activation),
            Dropout(rate=projection_dropout),
        ])
    projection_layers.extend([
        Dense(projection_dim, activation=None),
        Dropout(rate=projection_dropout),
    ])
    q1_encoded = time_distributed(q1_embed, projection_layers)
    q2_encoded = time_distributed(q2_embed, projection_layers)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compare
    q1_combined = Concatenate()(
        [q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()(
        [q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])
    compare_layers = [
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
        Dense(compare_dim, activation=activation),
        Dropout(compare_dropout),
    ]
    q1_compare = time_distributed(q1_combined, compare_layers)
    q2_compare = time_distributed(q2_combined, compare_layers)

    # # Aggregate
    # q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    # q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])


   
    q1_rep_max = MyMaxPool(axis=1)(q1_compare)
    q2_rep_max = MyMaxPool(axis=1)(q2_compare)


    cro_max = cross(q1_rep_max,q2_rep_max,compare_dim)
 
    dist = distence(q1_rep_max,q2_rep_max)
    

    #dense = cro
    dense = Concatenate()([
        q1_rep_max, q2_rep_max,cro_max,dist,
        ])

    #merged = Concatenate()([q1_rep, q2_rep,magic_dense])
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation=activation)(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[q1, q2,q1_w,q2_w,magic_input], outputs=out_)
    model.compile(optimizer=Adam(lr=lr), loss='binary_crossentropy',
                   metrics = [Precision,Recall,F1,])
    model.summary()

    return model


def esim(pretrained_embedding=config.word_embed_weights,
         maxlen=MAX_LEN,
         lstm_dim=300,
         dense_dim=300,
         dense_dropout=0.5):

    # Based on arXiv:1609.06038

    magic_input = Input(shape=(len(config.feats),))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)

    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))
    q1_w = Input(name='q1_w', shape=(maxlen,))
    q2_w = Input(name='q2_w', shape=(maxlen,))
    # Embedding
    embedding = create_pretrained_embedding(
        pretrained_embedding, mask_zero=False)
    bn = BatchNormalization(axis=2)
    q1_embed = bn(embedding(q1))
    q2_embed = bn(embedding(q2))

    # Encode
    encode = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))
    q1_encoded = encode(q1_embed)
    q2_encoded = encode(q2_embed)

    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)

    # Compose
    q1_combined = Concatenate()(
        [q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()(
        [q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

    compose = Bidirectional(CuDNNLSTM(lstm_dim, return_sequences=True))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)

    # Aggregate
    q1_rep = apply_multiple(q1_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])
    q2_rep = apply_multiple(q2_compare, [GlobalAvgPool1D(), GlobalMaxPool1D()])

    # Classifier
    cro = cross(q1_rep,q2_rep,lstm_dim*2)
    dist = distence(q1_rep,q2_rep)
    #dense = cro
    dense = Concatenate()([q1_rep, q2_rep])


    dense = BatchNormalization()(dense)
    dense = Dense(dense_dim, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(dense_dim, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)


    model = Model(inputs=[q1, q2,q1_w,q2_w,magic_input], outputs=out_)
    model.compile(loss='binary_crossentropy',
                  optimizer="adam", metrics = [Precision,Recall,F1,])
    model.summary()
    return model





def esim_blok(q1_encoded,q2_encoded,att_flag=True):
    lstm_dim = 300
   
    
    # Attention
    q1_aligned, q2_aligned = soft_attention_alignment(q1_encoded, q2_encoded)


     # Compose
    q1_combined = Concatenate()(
        [q1_encoded, q2_aligned, submult(q1_encoded, q2_aligned)])
    q2_combined = Concatenate()(
        [q2_encoded, q1_aligned, submult(q2_encoded, q1_aligned)])

    
    compose = Bidirectional(CuDNNGRU(lstm_dim, return_sequences=att_flag))
    q1_compare = compose(q1_combined)
    q2_compare = compose(q2_combined)


   
    return q1_compare ,q2_compare
def BMA_GRU(pretrained_embedding=config.word_embed_weights,
         maxlen=MAX_LEN,
         lstm_dim=300,
         dense_dim=300,
         dense_dropout=0.2,
         pool="max",
         mode='char+word'):

    # Based on arXiv:1609.06038

    magic_input = Input(shape=(len(config.feats),))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='elu')(magic_dense)

    q1 = Input(name='q1', shape=(maxlen,))
    q2 = Input(name='q2', shape=(maxlen,))

    q1_w = Input(name='q1_w', shape=(maxlen,))
    q2_w = Input(name='q2_w', shape=(maxlen,))

    # Embedding
    emb_layer = create_pretrained_embedding(
        config.char_embed_weights, mask_zero=False)
    emb_layer_word = create_pretrained_embedding(
        config.word_embed_weights, mask_zero=False)
    
    # Encode
    encode = Sequential()
    encode.add(emb_layer)
    encode.add(BatchNormalization(axis=2))
    encode.add(Bidirectional(CuDNNGRU(lstm_dim, return_sequences=True)))
    
    encode2 = Sequential()
    encode2.add(emb_layer_word)
    encode2.add(BatchNormalization(axis=2))
    encode2.add(Bidirectional(CuDNNGRU(lstm_dim, return_sequences=True)))

    q1_encoded = encode(q1)
    q2_encoded = encode(q2)

    q1_w_encoded = encode2(q1_w)
    q2_w_encoded = encode2(q2_w)

   
   

    att_flag=True
    q1_compare,q2_compare=esim_blok(q1_encoded,q2_encoded,att_flag)
    q1_compare_w,q2_compare_w=esim_blok(q1_w_encoded,q2_w_encoded,att_flag)

    # q1_rep ,q2_rep = q1_encoded,q2_encoded
    # q1_w_rep , q2_w_rep = q1_w_encoded,q2_w_encoded

    # q1_rep ,q2_rep = q1_compare,q2_compare
    # q1_w_rep , q2_w_rep = q1_compare_w,q2_compare_w

    


    if pool=='max':
        q1_rep = MyMaxPool(axis=1)(q1_compare)
        q2_rep = MyMaxPool(axis=1)(q2_compare)

        q1_w_rep = MyMaxPool(axis=1)(q1_compare_w)
        q2_w_rep = MyMaxPool(axis=1)(q2_compare_w)
    elif pool=='mean':

        q1_rep = MyMeanPool(axis=1)(q1_compare)
        q2_rep = MyMeanPool(axis=1)(q2_compare)

        q1_w_rep = MyMeanPool(axis=1)(q1_compare_w)
        q2_w_rep = MyMeanPool(axis=1)(q2_compare_w)
    else:
        q1_rep = Attention(maxlen)(q1_compare)
        q2_rep = Attention(maxlen)(q2_compare)

        q1_w_rep = Attention(maxlen)(q1_compare_w)
        q2_w_rep = Attention(maxlen)(q2_compare_w)



    # # Aggregate
    # q1_rep = apply_multiple(q1_compare, [MyMaxPool(axis=1), MyMeanPool(axis=1)])
    # q2_rep = apply_multiple(q2_compare, [MyMaxPool(axis=1), MyMeanPool(axis=1)])

   
    

    # Classifier
    cro = cross(q1_rep,q2_rep,lstm_dim*2)
    dist = distence(q1_rep,q2_rep)  
    dist2 = distence(q1_w_rep,q2_w_rep)
    #dense = cro
    
    if mode =="char":

        dense = Concatenate()([q1_rep, q2_rep,])
    elif mode =="word":
        dense = Concatenate()([q1_w_rep,q2_w_rep])
    else:
        dense = Concatenate()([q1_rep, q2_rep,q1_w_rep,q2_w_rep])

   
    dense = Dropout(dense_dropout)(dense)
    dense = Dense(dense_dim, activation='relu')(dense)
    dense = BatchNormalization()(dense)
    dense = Dropout(dense_dropout)(dense)
    out_ = Dense(1, activation='sigmoid')(dense)


    model = Model(inputs=[q1, q2,q1_w,q2_w,magic_input], outputs=out_)
    model.compile(loss='binary_crossentropy',
                  optimizer="adam", metrics = [Precision,Recall,F1,])
    model.summary()
    return model
