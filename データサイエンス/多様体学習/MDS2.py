# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:24:35 2019

@author: ShimaLab
"""


import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from mpl_toolkits import mplot3d


def random_projection(X,dimension=3,rseed=42):
    assert dimension >= X.shape[1]
    
    rng=np.random.RandomState(rseed)
    C=rng.randn(dimension,dimension)
    e,V=np.linalg.eigh(np.dot(C,C.T))
    return np.dot(X,V[:X.shape[1]])

X3=random_projection(X,3)

ax=plt.axes(projection='3d')
ax.scatter(X3[:,0],X3[:,1],X3[:,2],**colorize)
ax.view_init(azim=70,elev=50)

plt.figure()

model=MDS(n_components=2,random_state=1)
out3=model.fit_transform(X3)
plt.scatter(X3[:,0],X3[:,1],X3[:,2],**colorize)
plt.axis('equal')