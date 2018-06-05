from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda, Dense, Dropout
from keras.layers.recurrent import LSTM, GRU
from keras.layers.wrappers import Bidirectional
from keras.legacy.layers import Highway
from keras.layers import TimeDistributed
import keras.backend as K
from keras.layers.normalization import BatchNormalization


class WordRepresLayer(object):
    """Word embedding representation layer
    """
    def __init__(self, sequence_length, nb_words,
                 word_embedding_dim, embedding_matrix):
        self.model = Sequential()
        self.model.add(Embedding(nb_words,
                                 word_embedding_dim,
                                 weights=[embedding_matrix],
                                 input_length=sequence_length,
                                 trainable=False))

    def __call__(self, inputs):
        return self.model(inputs)


class CharRepresLayer(object):
    """Char embedding representation layer
    """
    def __init__(self, sequence_length, nb_chars, nb_per_word,
                 embedding_dim, rnn_dim, rnn_unit='gru', dropout=0.0):
        def _collapse_input(x, nb_per_word=0):
            x = K.reshape(x, (-1, nb_per_word))
            return x

        def _unroll_input(x, sequence_length=0, rnn_dim=0):
            x = K.reshape(x, (-1, sequence_length, rnn_dim))
            return x

        if rnn_unit == 'gru':
            rnn = GRU
        else:
            rnn = LSTM
        self.model = Sequential()
        self.model.add(Lambda(_collapse_input,
                              arguments={'nb_per_word': nb_per_word},
                              output_shape=(nb_per_word,),
                              input_shape=(sequence_length, nb_per_word,)))
        self.model.add(Embedding(nb_chars,
                                 embedding_dim,
                                 input_length=nb_per_word,
                                 trainable=True))
        self.model.add(rnn(rnn_dim,
                           dropout=dropout,
                           recurrent_dropout=dropout))
        self.model.add(Lambda(_unroll_input,
                              arguments={'sequence_length': sequence_length,
                                         'rnn_dim': rnn_dim},
                              output_shape=(sequence_length, rnn_dim)))
        
    def __call__(self, inputs):
        return self.model(inputs)


class ContextLayer(object):
    """Word context layer
    """
    def __init__(self, rnn_dim, rnn_unit='gru', input_shape=(0,),
                 dropout=0.0, highway=False, return_sequences=False,
                 dense_dim=0):
        if rnn_unit == 'gru':
            rnn = GRU
        else:
            rnn = LSTM
        self.model = Sequential()
        self.model.add(
            Bidirectional(rnn(rnn_dim,
                              dropout=dropout,
                              recurrent_dropout=dropout,
                              return_sequences=return_sequences),
                          input_shape=input_shape))
        # self.model.add(rnn(rnn_dim,
        #                    dropout=dropout,
        #                    recurrent_dropout=dropout,
        #                    return_sequences=return_sequences,
        #                    input_shape=input_shape))
        if highway:
            if return_sequences:
                self.model.add(TimeDistributed(Highway(activation='tanh')))
            else:
                self.model.add(Highway(activation='tanh'))

        if dense_dim > 0:
            self.model.add(TimeDistributed(Dense(dense_dim,
                                                 activation='relu')))
            self.model.add(TimeDistributed(Dropout(dropout)))
            self.model.add(TimeDistributed(BatchNormalization()))

    def __call__(self, inputs):
        return self.model(inputs)


class PredictLayer(object):
    """Prediction layer.
    """
    def __init__(self, dense_dim, input_dim=0,
                 dropout=0.0):
        self.model = Sequential()
        self.model.add(Dense(dense_dim,
                             activation='relu',
                             input_shape=(input_dim,)))
        self.model.add(Dropout(dropout))
        self.model.add(BatchNormalization())
        self.model.add(Dense(2, activation='sigmoid'))

    def __call__(self, inputs):
        return self.model(inputs)