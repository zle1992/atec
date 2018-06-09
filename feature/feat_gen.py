#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd



def magic1(train_in):
    # https://www.kaggle.com/jturkewitz/magic-features-0-03-gain
    import numpy as np
    import pandas as pd

    train_orig = train_in.copy()

    df1 = train_orig[['q1_cut']].copy()
    df2 = train_orig[['q2_cut']].copy()

    df2.rename(columns={'q2_cut': 'q1_cut'}, inplace=True)

    train_questions = df1.append(df2)

    train_questions.reset_index(inplace=True, drop=True)

    questions_dict = pd.Series(
        train_questions.index.values, index=train_questions.q1_cut.values).to_dict()
    train_cp = train_orig.copy()

    comb = train_cp

    comb['q1_hash'] = comb['q1_cut'].map(questions_dict)
    comb['q2_hash'] = comb['q2_cut'].map(questions_dict)

    q1_vc = comb.q1_hash.value_counts().to_dict()
    q2_vc = comb.q2_hash.value_counts().to_dict()

    def try_apply_dict(x, dict_to_apply):
        try:
            return dict_to_apply[x]
        except KeyError:
            return 0
    # map to frequency space
    comb['q1_freq'] = comb['q1_hash'].map(
        lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))
    comb['q2_freq'] = comb['q2_hash'].map(
        lambda x: try_apply_dict(x, q1_vc) + try_apply_dict(x, q2_vc))

    # Calculate derivative features

    comb['freq_mean'] = (comb['q1_freq'] + comb['q2_freq']) / 2
    comb['freq_cross'] = comb['q1_freq'] * comb['q2_freq']
    comb['q1_freq_sq'] = comb['q1_freq'] * comb['q1_freq']
    comb['q2_freq_sq'] = comb['q2_freq'] * comb['q2_freq']

    ret_cols = ['q1_freq', 'q2_freq', 'freq_mean',
                'freq_cross', 'q1_freq_sq', 'q2_freq_sq']

    return comb[ret_cols]


