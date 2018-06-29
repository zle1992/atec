import numpy as np
import pandas as pd
from keras.models import *
from keras.layers import *
from keras.optimizers import *
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


def cross(input_1, input_2, out_shape):

    diff = Lambda(lambda x: K.abs(
        x[0] - x[1]), output_shape=(out_shape,))([input_1, input_2])
    mul = Lambda(lambda x: x[0] * x[1],
                 output_shape=(out_shape,))([input_1, input_2])

    add = Lambda(lambda x: x[0] + x[1],
                 output_shape=(out_shape,))([input_1, input_2])
    # maximum = Maximum()([Multiply()([input_1,input_1]),Multiply()([input_2,input_2])])
    # minmum = Maximum()([-Multiply()([input_1,input_1]),-Multiply()([input_2,input_2])])
    out_ = Concatenate()([mul, diff, add])
    return out_


def cosine_similarity(x1, x2):
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


def compute_euclidean_match_score(l, r):
    denominator = 1. + K.sqrt(
        -2 * K.batch_dot(l, r, axes=[2, 2]) +
        K.expand_dims(K.sum(K.square(l), axis=2), 2) +
        K.expand_dims(K.sum(K.square(r), axis=2), 1)
    )
    denominator = K.maximum(denominator, K.epsilon())
    return 1. / denominator


def distence(input_1, input_2):
    malstm_distance = Lambda(lambda x: K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True)), output_shape=(
        1,))([input_1, input_2])
    #cos_distance = cosine_similarity(input_1, input_2)
    #euclidean = compute_euclidean_match_score(input_1, input_2)
    #out_ = Concatenate()([malstm_distance,cos_distance])#euclidean])
    return malstm_distance
