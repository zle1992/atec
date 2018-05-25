#!/usr/bin/python
# -*- coding: utf-8 -*-


batch_size = 128
number_classes = 2
w2v_vec_dim = 256
word_maxlen = 20


model_dir = '../model_dir'
jieba_dict = 'data/jieba_dict.txt'
origin_csv = 'data/atec_nlp_sim_train.csv'

data_augment = True
#data_augment = False

#cut_char_level = False
#最原始
# word_embed_weight = 'data/word_embed_weight_.npy'
# w2v_content_word_model = 'data/train_w2v.model'
# data_hdf = 'data/atec_nlp_sim_train.hdf'

#自定义词典+停用词过滤
# data_hdf = 'data/atec_nlp_sim_train2.hdf'
# word_embed_weight = 'data/word_embed_weight_2.npy'
# w2v_content_word_model = 'data/train_w2v2.model'



cut_char_level = True

#最原始
word_embed_weight = 'data/word_embed_weight_char.npy'
w2v_content_word_model = 'data/train_w2v_char.model'
data_hdf = 'data/atec_nlp_sim_train_char.hdf'

