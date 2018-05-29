#!/usr/bin/python
# -*- coding: utf-8 -*-


batch_size = 128
number_classes = 2
w2v_vec_dim = 300
word_maxlen = 40


model_dir = '../model_dir'
jieba_dict = 'data/jieba/jieba_dict.txt'
stopwords_path = 'data/jieba/stopwords.txt'
origin_csv = 'data/data/atec_nlp_sim_train.csv'

data_augment = True
#data_augment = False





#最原始
# use_pre_train = False
#cut_char_level = False
# word_embed_weight = 'data/word_embed_weight_.npy'
# w2v_content_word_model = 'data/train_w2v.model'
# data_hdf = 'data/atec_nlp_sim_train.hdf'

#自定义词典+停用词过滤


use_pre_train = False
cut_char_level = False
#data_hdf = 'data/atec_nlp_sim_train2.hdf'
data_hdf = 'data/atec_nlp_sim_train2_magic.hdf'
word_embed_weight = 'data/word_embed_weight_2.npy'
w2v_content_word_model = 'data/train_w2v2.model'


# char

# use_pre_train = False
# cut_char_level = True
# word_embed_weight = 'data/word_embed_weight_char.npy'
# w2v_content_word_model = 'data/train_w2v_char.model'
# data_hdf = 'data/atec_nlp_sim_train_char.hdf'



# use_pre_train = True
# cut_char_level = False
# word_embed_weight = 'data/pre_w2v/my_embeding.npy'
# word_embed_vocab = 'data/pre_w2v/my_vovab.npy'
# w2v_pre_train_dict = 'data/pre_w2v/sgns.zhihu.word'
# data_hdf = 'data/atec_nlp_sim_train_pre_train.hdf'