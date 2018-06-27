from __future__ import print_function
from keras import backend as K
#from keras.layers import Input, Convolution1D, Convolution2D, AveragePooling1D, GlobalAveragePooling1D, Dense, Lambda, merge, TimeDistributed, RepeatVector, Permute, ZeroPadding1D, ZeroPadding2D, Reshape, Dropout, BatchNormalization
from keras.models import Model
from keras.layers import *
from keras.optimizers import Nadam, Adam
import numpy as np
import sys
sys.path.append('utils/')
import config



def compute_cos_match_score(l_r):
    l, r = l_r
    return 1-K.batch_dot(
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






def ABCNN(
    left_seq_len, right_seq_len, nb_filter, filter_widths,
    depth=2, dropout=0.2, abcnn_1=True, abcnn_2=True, collect_sentence_representations=False, mode="euclidean", batch_normalize=True
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
 
    magic_input = Input(shape=(len(config.feats),))
    magic_dense = BatchNormalization()(magic_input)
    magic_dense = Dense(64, activation='relu')(magic_dense)

    left_input = Input(shape=(left_seq_len, ))
    right_input = Input(shape=(right_seq_len,))

    # Embedding
    pretrained_weights = np.load(config.word_embed_weight)
    in_dim, out_dim = pretrained_weights.shape



    embedding = Embedding(in_dim, out_dim, weights=[
                          pretrained_weights], trainable=False,)


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

        # concat attention
        # (samples, channels, rows, cols)
        left_embed = merge([left_embed, attention_left],
                           mode="concat", concat_axis=1)
        right_embed = merge([right_embed, attention_right],
                            mode="concat", concat_axis=1)

        # # Padding so we have wide convolution
        left_embed_padded = ZeroPadding2D((filter_width - 1, 0))(left_embed)
        right_embed_padded = ZeroPadding2D((filter_width - 1, 0))(right_embed)

       
        # left_embed_padded = left_embed
        # right_embed_padded = right_embed

        # 2D convolutions so we have the ability to treat channels.
        # Effectively, we are still doing 1-D convolutions.
  
        my_conv2d = Conv2D( data_format="channels_first", padding="valid", filters=nb_filter, kernel_size=(filter_width, out_dim))
        my_conv2d2 = Conv2D( data_format="channels_first", padding="valid", filters=nb_filter, kernel_size=(filter_width, out_dim))
        conv_left = my_conv2d(left_embed_padded)
        
        # Reshape and Permute to get back to 1-D
        conv_left = (Reshape((conv_left._keras_shape[1], conv_left._keras_shape[2])))(
            conv_left)
        conv_left = Permute((2, 1))(conv_left)


        conv_right = my_conv2d2(right_embed_padded)

        # Reshape and Permute to get back to 1-D
        conv_right = (
            Reshape((conv_right._keras_shape[1], conv_right._keras_shape[2])))(conv_right)
        conv_right = Permute((2, 1))(conv_right)

    else:
        # Padding so we have wide convolution
        left_embed_padded = ZeroPadding1D(filter_width - 5)(left_embed)
        right_embed_padded = ZeroPadding1D(filter_width - 5)(right_embed)
        conv_left = Convolution1D(
            nb_filter, filter_width, activation="tanh", border_mode="valid")(left_embed_padded)
        conv_right = Convolution1D(
            nb_filter, filter_width, activation="tanh", border_mode="valid")(right_embed_padded)

    
        conv_left = BatchNormalization()(conv_left)
        conv_right = BatchNormalization()(conv_right)
    
    conv_left = Dropout(dropout)(conv_left)
    conv_right = Dropout(dropout)(conv_right)

    pool_left = AveragePooling1D(
        pool_length=filter_width, stride=1, border_mode="valid")(conv_left)
    pool_right = AveragePooling1D(
        pool_length=filter_width, stride=1, border_mode="valid")(conv_right)

    pool_left = ZeroPadding1D(filter_width - 1)(pool_left)
    pool_right = ZeroPadding1D(filter_width - 1)(pool_right)
    assert pool_left._keras_shape[1] == left_seq_len, "%s != %s" % (
        pool_left._keras_shape[1], left_seq_len)
    assert pool_right._keras_shape[1] == right_seq_len, "%s != %s" % (
        pool_right._keras_shape[1], right_seq_len)

    if collect_sentence_representations or depth == 1:  # always collect last layers global representation
        left_sentence_representations.append(
            GlobalAveragePooling1D()(conv_left))
        right_sentence_representations.append(
            GlobalAveragePooling1D()(conv_right))

    # ###################### #
    # ### END OF ABCNN-1 ### #
    # ###################### #

    for i in range(depth - 1):
        filter_width = filter_widths.pop(0)
        pool_left = ZeroPadding1D(filter_width - 1)(pool_left)
        pool_right = ZeroPadding1D(filter_width - 1)(pool_right)
        # Wide convolution
        conv_left = Convolution1D(
            nb_filter, filter_width, activation="tanh", border_mode="valid")(pool_left)
        conv_right = Convolution1D(
            nb_filter, filter_width, activation="tanh", border_mode="valid")(pool_right)

        if abcnn_2:
            conv_match_score = MatchScore(conv_left, conv_right, mode=mode)

            # compute attention
            conv_attention_left = Lambda(lambda match: K.sum(
                match, axis=-1), output_shape=(conv_match_score._keras_shape[1],))(conv_match_score)
            conv_attention_right = Lambda(lambda match: K.sum(
                match, axis=-2), output_shape=(conv_match_score._keras_shape[2],))(conv_match_score)

            conv_attention_left = Permute((2, 1))(
                RepeatVector(nb_filter)(conv_attention_left))
            conv_attention_right = Permute((2, 1))(
                RepeatVector(nb_filter)(conv_attention_right))

            # apply attention  TODO is "multiply each value by the sum of it's
            # respective attention row/column" correct?
            conv_left = merge([conv_left, conv_attention_left], mode="mul")
            conv_right = merge([conv_right, conv_attention_right], mode="mul")

       
            conv_left = BatchNormalization()(conv_left)
            conv_right = BatchNormalization()(conv_right)

        conv_left = Dropout(dropout)(conv_left)
        conv_right = Dropout(dropout)(conv_right)

        pool_left = AveragePooling1D(
            pool_length=filter_width, stride=1, border_mode="valid")(conv_left)
        pool_right = AveragePooling1D(
            pool_length=filter_width, stride=1, border_mode="valid")(conv_right)

        assert pool_left._keras_shape[1] == left_seq_len
        assert pool_right._keras_shape[1] == right_seq_len

        # always collect last layers global representation
        if collect_sentence_representations or (i == (depth - 2)):
            left_sentence_representations.append(
                GlobalAveragePooling1D()(conv_left))
            right_sentence_representations.append(
                GlobalAveragePooling1D()(conv_right))

    # ###################### #
    # ### END OF ABCNN-2 ### #
    # ###################### #

    # Merge collected sentence representations if necessary
    left_sentence_rep = left_sentence_representations.pop(-1)
    if left_sentence_representations:
        left_sentence_rep = merge(
            [left_sentence_rep] + left_sentence_representations, mode="concat")

    right_sentence_rep = right_sentence_representations.pop(-1)
    if right_sentence_representations:
        right_sentence_rep = merge(
            [right_sentence_rep] + right_sentence_representations, mode="concat")

    global_representation = merge(
        [left_sentence_rep, right_sentence_rep], mode="concat")
    global_representation = Dropout(dropout)(global_representation)

    diff_mul =True
    if diff_mul:
        # Add logistic regression on top.
        diff = Lambda(lambda x: K.abs(
            x[0] - x[1]), output_shape=(left_sentence_rep._keras_shape[1],))([left_sentence_rep, right_sentence_rep])
        mul = Lambda(lambda x: x[0] * x[1],
                 output_shape=(left_sentence_rep._keras_shape[1],))([left_sentence_rep, right_sentence_rep])
        global_representation =  merge([diff,mul,magic_dense],mode="concat")
    else:
        global_representation =  merge([magic_dense,global_representation],mode="concat")
    classify = Dense(2, activation="sigmoid")(global_representation)

    model = Model([left_input, right_input,magic_input], output=classify)
    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(), metrics=['acc'])
    model.summary()
    return model