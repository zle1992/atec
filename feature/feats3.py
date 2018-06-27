# coding: utf-8


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

def mytfidf(df_data,cols=['q1_cut','q2_cut']):
    tfidf = TfidfVectorizer( ngram_range=(1, 1))

    questions_txt = pd.Series(
        df_data[cols[0]].tolist() +
        df_data[cols[1]].tolist()
    ).astype(str)

    tfidf.fit_transform(questions_txt)


    tfidf_sum1 = []
    tfidf_sum2 = []
    tfidf_mean1 = []
    tfidf_mean2 = []
    tfidf_len1= []
    tfidf_len2 = []

    for index, row in df_data.iterrows():
        tfidf_q1 = tfidf.transform([str(row.question1)]).data
        tfidf_q2 = tfidf.transform([str(row.question2)]).data
        
        tfidf_sum1.append(np.sum(tfidf_q1))
        tfidf_sum2.append(np.sum(tfidf_q2))
        tfidf_mean1.append(np.mean(tfidf_q1))
        tfidf_mean2.append(np.mean(tfidf_q2))
        tfidf_len1.append(len(tfidf_q1))
        tfidf_len2.append(len(tfidf_q2))

    df_feat['tfidf_sum1'] = tfidf_sum1
    df_feat['tfidf_sum2'] = tfidf_sum2
    df_feat['tfidf_mean1'] = tfidf_mean1
    df_feat['tfidf_mean2'] = tfidf_mean2
    df_feat['tfidf_len1'] = tfidf_len1
    df_feat['tfidf_len2'] = tfidf_len2


    df_feat.fillna(0.0)
    return df_feat