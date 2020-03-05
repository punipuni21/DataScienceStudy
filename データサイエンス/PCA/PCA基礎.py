# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 08:49:47 2019

@author: ShimaLab
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

from sklearn.decomposition import PCA


rng=np.random.RandomState(1)
X=np.dot(rng.rand(2,2),rng.randn(2,200)).T
plt.scatter(X[:,0],X[:,1])
plt.axis('equal')


pca=PCA(n_components=2)
pca.fit(X)
print(pca.components_)
print(pca.explained_variance_)

#可視化
def draw_vector(v0,v1,ax=None):
    ax=ax or plt.gca()
    arrowprops=dict(arrowstyle='->',linewidth=2,shrinkA=0,shrinkB=0)
    ax.annotate('',v1,v0,arrowprops=arrowprops)

plt.scatter(X[:,0],X[:,1],alpha=0.2)
for length,vector in zip(pca.explained_variance_,pca.components_):
    v=vector*3*np.sqrt(length)
    draw_vector(pca.mean_,pca.mean_+v)
    
plt.axis('equal')

#　次元削減

pca=PCA(n_components=1)
pca.fit(X)
#1次元に圧縮
X_pca=pca.transform(X)
print("oroginal shape:",X.shape)
print("transformed shape:",X_pca.shape)


X_new=pca.inverse_transform(X_pca)
plt.scatter(X[:,0],X[:,1])
plt.scatter(X_new[:,0],X_new[:,1],alpha=0.8)
plt.axis('equal')
