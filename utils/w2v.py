# -*- coding: utf-8 -*-
import os
import sys
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

sys.path.append('utils/')
#sys.path.append('feature/')
import config

from CutWord import cut_word,read_cut

def read_words(data):
    words = list(data.q1_cut) + list(data.q2_cut)
    words_all = []
    for word in words:
        words_all.extend(word)
    words_set = list(set(words_all))
    words_set = [u'unknow'] + words_set
    words_set = words_set + [u'pos', u'eos', u'padding']
    words_set = np.array(words_set)
    return words_set


def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
    lines_num, dim = 0, 0
    vectors = {}
    iw = []
    wi = {}
    with open(path, 'rb') as f:
        first_line = True
        for line in f:
            if first_line:
                first_line = False
                dim = int(line.rstrip().split()[1])
                continue
            lines_num += 1
            tokens = line.rstrip().split(' ')
            vectors[tokens[0]] = np.asarray([float(x) for x in tokens[1:]])
            iw.append(tokens[0])
            if topn != 0 and lines_num >= topn:
                break
    for i, w in enumerate(iw):
        wi[w] = i
    return vectors, iw, wi, dim


def load_pre_train_embeddings(vocab, vectors,):
    vector_length = len(vocab)
    weights = np.zeros((vector_length, 300),   dtype='float32')
    cnt = 0
    # Normalization
    for index, word in enumerate(vocab):
        if word.encode('utf8') in vectors:
            weights[index] = vectors[word.encode('utf8')]
        else:
            weights[index] = np.random.random(size=weights.shape[1])
            cnt += 1
    print('vocab oov:{0}/{1}'.format(cnt, len(vocab)))
    return weights


def save_my_w2v(path):
    data = read_cut(path)
    vocab = read_words(data)
    # Read top n word vectors. Read all vectors when topn is 0
    vectors, iw, wi, dim = read_vectors('data/pre_w2v/sgns.zhihu.word', 0)

    m = load_pre_train_embeddings(vocab, vectors)
    np.save(config.word_embed_vocab, vocab)
    np.save(config.word_embed_weight, m)


def load_pre_train_w2v(path):
    if not os.path.exists(config.word_embed_vocab):
        save_my_w2v(path)
    vocab = np.load(config.word_embed_vocab)
    vocab = {w: i for i, w in enumerate(vocab)}
    embed_weights = np.load(config.word_embed_weight)

    print('load embed_weights and vocab!')
    return vocab, embed_weights




#######################################
def make_w2v(path):
    if not os.path.exists(config.w2v_content_word_model):

        data = cut_word(config.origin_csv,config.cut_char_level)
        content = list(data.q1_cut) + list(data.q2_cut)
        model = Word2Vec(content, size=config.w2v_vec_dim, window=5, min_count=5,
                         )
        model.save(config.w2v_content_word_model)

    model = Word2Vec.load(config.w2v_content_word_model)

    weights = model.wv.syn0
    vocab = dict([(k, v.index + 1) for k, v in model.wv.vocab.items()])
    vocab['<-UNKNOW->'] = len(vocab) + 1
    embed_weights = np.zeros(shape=(weights.shape[0] + 2, weights.shape[1]))
    embed_weights[1:weights.shape[0] + 1] = weights
    unk_vec = np.random.random(size=weights.shape[1]) * 0.5
    pading_vec = np.random.random(size=weights.shape[1]) * 0
    embed_weights[weights.shape[0] + 1] = unk_vec - unk_vec.mean()
    embed_weights[0] = pading_vec

    np.save(config.word_embed_weight, embed_weights)
    print(embed_weights.shape)
    print('save embed_weights!')
    return vocab, embed_weights

def load_my_train_w2v(path):
    return make_w2v(path)
################################################



if __name__ == '__main__':

    #save_my_w2v(path=config.origin_csv)
    make_w2v(path=config.origin_csv)
