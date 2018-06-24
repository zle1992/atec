#!/usr/bin/python
# -*- coding: utf-8 -*-


batch_size = 128
number_classes = 2
w2v_vec_dim = 256
word_maxlen = 40


model_dir = '../model_dir'
jieba_dict = 'data/share/jieba/jieba_dict.txt'
stopwords_path = 'data/share/jieba/stops.txt'
origin_csv = 'data/data/atec_nlp_sim_train2.csv'

data_augment = True
shuffer = True


nofeats = False
if nofeats:
    feats = [u'pading1', u'pading2']
else:
    feats = [u'q1_freq', u'q2_freq', u'freq_mean', u'freq_cross', u'q1_freq_sq',
             u'q2_freq_sq',
             u'len_diff',
             u'shingle_similarity_1',
             u'shingle_similarity_2',
             u'shingle_similarity_3',
             u'common_words',
             'cwc_min',
             'cwc_max',
             'csc_min',
             'csc_max',
             'ctc_min',
             'ctc_max',
             'last_word_eq',
             'first_word_eq',
             'abs_len_diff',
             'mean_len',
             'token_set_ratio',
             'token_sort_ratio',
             'fuzz_ratio',
             'fuzz_partial_ratio',
             'longest_substr_ratio']


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


use_pre_train = False


cut_char_level = True

if cut_char_level:

    data_cut_hdf = 'data/cache/train_cut_char.hdf'
    train_feats = 'data/cache/train_feats_char.hdf'
    data_feat_hdf = 'data/cache/train_magic_char.hdf'
    train_df = 'data/cache/train_magic_char_train_f{0}.hdf'.format(len(feats))
    dev_df = 'data/cache/train_magic_char_more_dev_f{0}.hdf'.format(len(feats))

    word_embed_weight = 'data/share/my_w2v/word_embed_weight_char.npy'
    w2v_content_word_model = 'data/share/my_w2v/train_char.model'
else:
    data_cut_hdf = 'data/cache/train_cut_word.hdf'
    train_feats = 'data/cache/train_feats_word.hdf'
    data_feat_hdf = 'data/cache/train_magic_word.hdf'
    train_df = 'data/cache/train_magic_word_train_f{0}.hdf'.format(len(feats))
    dev_df = 'data/cache/train_magic_word_more_dev_f{0}.hdf'.format(len(feats))

    word_embed_weight = 'data/share/my_w2v/word_embed_weight_word.npy'
    w2v_content_word_model = 'data/share/my_w2v/train_word.model'


if cut_char_level:
    stack_path = 'data/share/stack/char_'
else:
    stack_path = 'data/share/stack/word_'


train_featdires = ['data/feats0_train.csv',
                  'data/feats1_train.csv',
                  'data/feats2_train.csv',
                  ]

test_featdires = ['data/share/feats0_test.csv',
                  'data/share/feats1_test.csv',
                  'data/share/feats2_test.csv']
