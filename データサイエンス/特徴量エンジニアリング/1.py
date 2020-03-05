# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 13:44:38 2019

@author: ShimaLab
"""

from sklearn.feature_extraction import DictVectorizer

data=[
      {'price':850000,'room':4,'neighborhood':'Queen Anne'},
      {'price':700000,'room':3,'neighborhood':'Fremont'},
      {'price':650000,'room':3,'neighborhood':'Wallingford'},
      {'price':600000,'room':2,'neighborhood':'Fremont'},]

#ワンホットエンコーディングを行う
vec = DictVectorizer(sparse=False,dtype=int)
data2 = vec.fit_transform(data)

#特徴名を調べる
data3 = vec.get_feature_names()

vec2 = DictVectorizer(sparse=False,dtype=int)
data4 = vec.fit_transform(data)
