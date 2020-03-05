# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:05:37 2019

@author: ShimaLab
"""

import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans

X,ytrue=make_blobs(n_samples=300,centers=4,cluster_std=0.60,random_state=0)
plt.scatter(X[:,0],X[:,1],s=50)

plt.figure()
kmeans=KMeans(n_clusters=4)
kmeans.fit(X)
ykmeans=kmeans.predict(X)
plt.scatter(X[:,0],X[:,1],s=50,c=ykmeans,cmap='viridis')

centers=kmeans.cluster_centers_
plt.scatter(centers[:,0],centers[:,1],c='black',s=200,alpha=0.5)