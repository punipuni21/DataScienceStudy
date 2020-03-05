# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 09:48:08 2019

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
from sklearn.datasets import load_digits
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

digits=load_digits()

kmeans=KMeans(n_clusters=10,random_state=0)
cluster=kmeans.fit_predict(digits.data)


fig,ax=plt.subplots(2,5,figsize=(8,3))
centers=kmeans.cluster_centers_.reshape(10,8,8)
for axi,center in zip(ax.flat,centers):
    axi.set(xticks=[],yticks=[])
    axi.imshow(center,interpolation='nearest',cmap=plt.cm.binary)
    
labels=np.zeros_like(cluster)

for i in range(10):
    mask=(cluster==i)
    labels[mask]=mode(digits.target[mask])[0]
    
score = accuracy_score(digits.target,labels)

plt.figure()

mat = confusion_matrix(digits.target,labels)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,cmap='RdPu',xticklabels=digits.target_names,yticklabels=digits.target_names)

tsne=TSNE(n_components=2,init='pca',random_state=0)
digits_proj=tsne.fit_transform(digits.data)

kmeans=KMeans(n_clusters=10,random_state=0)
clusters=kmeans.fit_predict(digits_proj)

labels=np.zeros_like(clusters)
for i in range(10):
    mask=(clusters==i)
    labels[mask]=mode(digits.target[mask])[0]
    
score2 = accuracy_score(digits.target,labels)
    
    
    
    
    