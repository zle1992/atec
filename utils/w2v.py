#/usr/bin/env python
# coding=utf-8
import os
import sys
import numpy as np
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import pandas as pd
import pandas as pd
import jieba
sys.path.append('utils/')
import config

jieba.load_userdict(config.jieba_dict)


#stopwords = [line.strip() for line in open(filepath,'r',encoding='utf-8').readlines()]
stopwords = ['?', ',', '。']


def cut_single(x):
    res = []
    if config.cut_char_level:
        setence_seged = list(x.strip())
    else:
        setence_seged = jieba.cut(x.strip())
    for word in setence_seged:
        if word not in stopwords:
            res.append(word)
    return res


def make_w2v(path):
    if not os.path.exists(config.w2v_content_word_model):

        data = read_cut(path)
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


def load_pre_train_w2v(path):
    def read_vectors(path, topn):  # read top n word vectors, i.e. top is 10000
        lines_num, dim = 0, 0
        vectors = {}
        iw = []
        wi = {}
        with open(path,'rb') as f:
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
    def load_glove_embeddings(vocab, vectors,n_unknown=-1):
        max_vector_length = len(vocab) + 1  # index start from 1
        matrix = np.zeros((max_vector_length 2, 300), dtype='float32')  # 2 for <PAD> and <EOS>
        # Normalization
        for word in vocab:
            if word in vectors:
                matrix[vocab[word]] =vectors[word]

        return matrix

    if not os.path.exists(config.w2v_pre_train_word_model):

        data = read_cut(path)
        words = list(data.q1_cut) + list(data.q2_cut)
        words_all = []
        for word in words:
            words_all.extend(word)
        word_all = set(word_all)
        vocab = dict([(word, id + 1) for id, word in words_all])
    vocab['<-UNKNOW->'] = len(vocab) + 1
    vectors, iw, wi, dim = read_vectors('../data/pre_w2v/sgns.zhihu.word', -1)

    np.save(config.word_embed_weight, embed_weights)
    print(embed_weights.shape)
    print('save embed_weights!')
    return vocab, embed_weights


def padding_id(ids, padding_token=0, padding_length=None):
    if len(ids) > padding_length:
        return ids[:padding_length]
    else:
        return ids + [padding_token] * (padding_length - len(ids))


def word2id(contents, word_voc):
    ''' contents  list
    '''
#     contents = str(contents)
#     contents = contents.split()

    ids = [word_voc[c] if c in word_voc else len(word_voc) for c in contents]

    return padding_id(ids, padding_token=0, padding_length=config.word_maxlen)


def read_cut(path):

    data = pd.read_csv(path, sep='\t', encoding='utf8',
                       names=['id', 'q1', 'q2', 'label'])
    data['q1_cut'] = data['q1'].map(lambda x: cut_single(x))
    data['q2_cut'] = data['q2'].map(lambda x: cut_single(x))
    print('cut done')
    return data


def read_hdf(path):
    if not os.path.exists(config.data_hdf):
        data = read_data(path)
        data.to_hdf(config.data_hdf, "data")
    else:
        data = pd.read_hdf(config.data_hdf)
    return data


def read_data(path):
    data = read_cut(path)
    vocab, embed_weights = make_w2v(config.origin_csv)

    data['q1_cut_id'] = data['q1_cut'].map(lambda x: word2id(x, vocab))
    data['q2_cut_id'] = data['q2_cut'].map(lambda x: word2id(x, vocab))
    return data


if __name__ == '__main__':
    path = config.origin_csv
    read_data(path)
