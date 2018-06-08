#!/usr/bin/python
# -*- coding: utf-8 -*-


batch_size = 128
number_classes = 2
w2v_vec_dim = 256
word_maxlen = 40




 
                
model_dir = '../model_dir'
jieba_dict = 'data/jieba/jieba_dict.txt'
stopwords_path = 'data/jieba/aa.txt'
origin_csv = 'data/data/atec_nlp_sim_train.csv'

data_augment = True
more_data=0
#data_augment = False





#最原始
# use_pre_train = False
#cut_char_level = False
# word_embed_weight = 'data/word_embed_weight_.npy'
# w2v_content_word_model = 'data/train_w2v.model'
# data_hdf = 'data/atec_nlp_sim_train.hdf'

#自定义词典+停用词过滤


# use_pre_train = False
# cut_char_level = False
# data_hdf = 'data/atec_nlp_sim_train2.hdf'
# data_cut_hdf ='data/atec_cut.hdf'
# data_feat_hdf = 'data/atec_nlp_sim_train2_magic.hdf'
# word_embed_weight = 'data/word_embed_weight_2.npy'
# w2v_content_word_model = 'data/train_w2v2.model'


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


feats =[
#  u'q1_q2_intersect',
#        u'wordmatch1',

# 'bin_dist1',
#  'bin_dist2',
#  'diff1',
#  'diff2',
#  'diff_norm1',
#  'diff_norm2',
#  'diff_uni1',
#  'diff_uni2',

#  'inter_uni_r1',
#  'inter_uni_r2',
#  'intersect_r1',
#  'intersect_r2',
#  'jaccard_dist1',
#  'jaccard_dist2',

#  'len_diff',
#  'masi_dist1',
#  'masi_dist2',
#  'max1',
#  'max2',
#  'min1',
#  'min2',


#  'q1_len',
# 'q1_q2_intersect',
#  'q1_sum1',
#  'q1_sum2',
#  'q1_uni1',
#  'q1_uni2',


#  'q2_len',
#  'q2_sum1',
#  'q2_sum2',
#  'q2_uni1',
#  'q2_uni2',
 
]

# use_pre_train = False
# cut_char_level = False
# data_cut_hdf ='data/cache/train_cut_word.hdf'
# data_feat_hdf = 'data/cache/train_magic_word.hdf'
# train_df= 'data/cache/train_magic_word_train_f{0}.hdf'.format(len(feats))
# dev_df = 'data/cache/train_magic_word_more_dev_f{0}.hdf'.format(len(feats))

# word_embed_weight = 'data/my_w2v/word_embed_weight_word.npy'
# w2v_content_word_model = 'data/my_w2v/train_word.model'

use_pre_train = False
cut_char_level = True
data_cut_hdf ='data/cache/train_cut_char.hdf'
data_feat_hdf = 'data/cache/train_magic_char.hdf'
train_df= 'data/cache/train_magic_char_train_f{0}.hdf'.format(len(feats))
dev_df = 'data/cache/train_magic_char_more_dev_f{0}.hdf'.format(len(feats))

word_embed_weight = 'data/my_w2v/word_embed_weight_char.npy'
w2v_content_word_model = 'data/my_w2v/train_char.model'




















