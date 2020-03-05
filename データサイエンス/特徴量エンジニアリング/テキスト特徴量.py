# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:05:07 2019

@author: ShimaLab
"""
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

sample = ['problem of evil',
          'evil queen',
          'horizon problem']

vec = CountVectorizer()
X = vec.fit_transform(sample)


data5 = pd.DataFrame(X.toarray(),columns=vec.get_feature_names())

vec2 = TfidfVectorizer()
X = vec2.fit_transform(sample)
data6 = pd.DataFrame(X.toarray(),columns=vec.get_feature_names())