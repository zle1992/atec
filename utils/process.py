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
import w2v

jieba.load_userdict(config.jieba_dict)
stopwords = [line.strip() for line in open(config.stopwords_path, 'r').readlines()]
stopwords = [w.decode('utf8') for w in stopwords]
#stopwords = [u'？', u'。', u'吗',u'，',u'的']


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
    if not os.path.exists(config.word_embed_vocab):
        w2v.save_my_w2v(path)
    vocab = np.load(config.word_embed_vocab)
    vocab = {w: i for i, w in enumerate(vocab)}
    embed_weights = np.load(config.word_embed_weight)

    print('load embed_weights and vocab!')
    return vocab, embed_weights
#######################################


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
    if config.use_pre_train:
        vocab, embed_weights = load_pre_train_w2v(path)
    else:
        vocab, embed_weights = make_w2v(path)

    data['q1_cut_id'] = data['q1_cut'].map(lambda x: word2id(x, vocab))
    data['q2_cut_id'] = data['q2_cut'].map(lambda x: word2id(x, vocab))
    return data


if __name__ == '__main__':
    path = config.origin_csv
    #read_data(path)
    #trans_pre_train_w2v(path)
