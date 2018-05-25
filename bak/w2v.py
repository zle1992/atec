#!/usr/bin/python
# -*- coding: utf-8 -*-
##########################################################################
#
# Copyright (c) 2018 Baidu.com, Inc. All Rights Reserved
#
##########################################################################
"""


Authors: zhangle11(zhangle11@baidu.com)
Date:    2018-04-27 11:15:25

"""
import os

import sys
import numpy as np
import pickle
import multiprocessing
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import yaml
import pandas as pd
import jieba
config = yaml.load(open('config.yaml', 'r'))
data = pd.read_csv('data/atec_nlp_sim_train.csv', sep='\\t',
                   names=['id', 'q1', 'q2', 'label'])
