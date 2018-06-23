#!/usr/bin/python
# -*- coding: utf-8 -*-


import sys
import os
import pandas as pd
import numpy as np


sys.path.append('utils/')
sys.path.append('feature/fuzzywuzzy-master/')
import config
from CutWord import read_cut
from fuzzywuzzy import fuzz


stopwords = [line.strip() for line in open(config.stopwords_path, 'r').readlines()]
stopwords = [w.decode('utf8') for w in stopwords]
STOP_WORDS = stopwords
SAFE_DIV = 0.0001




def len_diff(s1, s2):
    return abs(len(s1) - len(s2))


def len_diff_ratio(s1, s2):
    return 2 * abs(len(s1) - len(s2)) / (len(s1) + len(s2))


def shingle_similarity(s1, s2, size=1):
    """Shingle similarity of two sentences."""
    def get_shingles(text, size):
        shingles = set()
        for i in range(0, len(text) - size + 1):
            shingles.add(text[i:i + size])
        return shingles

    def jaccard(set1, set2):
        x = len(set1.intersection(set2))
        y = len(set1.union(set2))
        return x, y

    x, y = jaccard(get_shingles(s1, size), get_shingles(s2, size))
    return x / float(y) if (y > 0 and x > 2) else 0.0


def common_words(s1, s2):
    s1_common_cnt = len([w for w in s1 if w in s2])
    s2_common_cnt = len([w for w in s2 if w in s1])
    return (s1_common_cnt + s2_common_cnt) / (len(s1) + len(s2))


def tf_idf():
    pass


def wmd():
    pass


def feats1_gen(data, cols=['q1', 'q2']):
    print('make feats1-------')
    data['len_diff'] = data.apply(
        lambda x: len_diff(x[cols[0]], x[cols[1]]), axis=1)
    data['shingle_similarity_1'] = data.apply(
        lambda x: shingle_similarity(x[cols[0]], x[cols[1]]), axis=1)
    data['shingle_similarity_2'] = data.apply(
        lambda x: shingle_similarity(x[cols[0]], x[cols[1]]), axis=1)
    data['shingle_similarity_3'] = data.apply(
        lambda x: shingle_similarity(x[cols[0]], x[cols[1]]), axis=1)
    data['common_words'] = data.apply(
        lambda x: common_words(x[cols[0]], x[cols[1]]), axis=1)
    return data


from array import array
def lcsubstrings(seq1, seq2, positions=False):
    """Find the longest common substring(s) in the sequences `seq1` and `seq2`.

    If positions evaluates to `True` only their positions will be returned,
    together with their length, in a tuple:

        (length, [(start pos in seq1, start pos in seq2)..])

    Otherwise, the substrings themselves will be returned, in a set.

    Example:

        >>> lcsubstrings("sedentar", "dentist")
        {'dent'}
        >>> lcsubstrings("sedentar", "dentist", positions=True)
        (4, [(2, 0)])
    """
    L1, L2 = len(seq1), len(seq2)
    ms = []
    mlen = last = 0
    if L1 < L2:
        seq1, seq2 = seq2, seq1
        L1, L2 = L2, L1

    column = array('L', range(L2))

    for i in range(L1):
        for j in range(L2):
            old = column[j]
            if seq1[i] == seq2[j]:
                if i == 0 or j == 0:
                    column[j] = 1
                else:
                    column[j] = last + 1
                if column[j] > mlen:
                    mlen = column[j]
                    ms = [(i, j)]
                elif column[j] == mlen:
                    ms.append((i, j))
            else:
                column[j] = 0
            last = old

    if positions:
        return (mlen, tuple((i - mlen + 1, j - mlen + 1) for i, j in ms if ms))
    return set(seq1[i - mlen + 1:i + 1] for i, _ in ms if ms)

def get_token_features(q1, q2):
    token_features = [0.0]*10

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    token_features[9] = (len(q1_tokens) + len(q2_tokens))/2
    return token_features


def get_longest_substr_ratio(a, b):
    strs = list(lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)

def extract_features(df,cols=['q1','q2']):
    print("Extracting features for train:")
    df["question1"] = df[cols[0]].fillna("")
    df["question2"] = df[cols[1]].fillna("")

    print("token features...")
    token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
    df["cwc_min"]       = list(map(lambda x: x[0], token_features))
    df["cwc_max"]       = list(map(lambda x: x[1], token_features))
    df["csc_min"]       = list(map(lambda x: x[2], token_features))
    df["csc_max"]       = list(map(lambda x: x[3], token_features))
    df["ctc_min"]       = list(map(lambda x: x[4], token_features))
    df["ctc_max"]       = list(map(lambda x: x[5], token_features))
    df["last_word_eq"]  = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    df["abs_len_diff"]  = list(map(lambda x: x[8], token_features))
    df["mean_len"]      = list(map(lambda x: x[9], token_features))

    print("fuzzy features..")
    df["token_set_ratio"]       = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    df["token_sort_ratio"]      = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"]            = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"]    = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    df["longest_substr_ratio"]  = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
    return df




# def save(data, out_path):
#     data = feats1_gen(data)
#     feats1 = ['len_diff',
#               'shingle_similarity_1',
#               'shingle_similarity_2',
#               'shingle_similarity_3',
#               'common_words']

#     data[['id'] + feats1].to_csv(out_path,index=False)


# if __name__ == '__main__':
#     path = config.origin_csv
#     data = read_cut(path)  # cut word
#     data = data_2id(data)  # 2id
#     out_path = 'data/cache/feats/feats1_train.csv'
#     save(data, out_path)
