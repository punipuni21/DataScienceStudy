# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:15:29 2019

@author: ShimaLab
"""


import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering

def find_clusters(X,n_clusters,rseed=2):
    rng=np.random.RandomState(rseed)
    i=rng.permutation(X.shape[0])[:n_clusters]
    centers=X[i]
    
    while True:
        labels=pairwise_distances_argmin(X,centers)
        
        new_centers=np.array([X[labels==i].mean(0)for i in range(n_clusters)])
        
        if np.all(centers==new_centers):
            break
        centers=new_centers
    return centers,labels

centers,labels=find_clusters(X,4)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')


plt.figure()
X,y=make_moons(200,noise=.05,random_state=0)
labels=KMeans(n_clusters=2,random_state=0).fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')


plt.figure()
model=SpectralClustering(n_clusters=2,affinity='nearest_neighbors',assign_labels='kmeans')
labels=model.fit_predict(X)
plt.scatter(X[:,0],X[:,1],c=labels,s=50,cmap='viridis')




