# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:08:55 2019

@author: ShimaLab
"""


import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS


def rotate(X,angle):
    theta=np.deg2rad(angle)
    R=[[np.cos(theta),np.sin(theta)],
        [-np.sin(theta),np.cos(theta)]]
    return np.dot(X,R)

X2=rotate(X,20)+5
plt.scatter(X2[:,0],X2[:,1],**colorize)
plt.axis('equal')

D=pairwise_distances(X)

plt.imshow(D,zorder=2,cmap='Blues',interpolation='nearest')
plt.colorbar()


D2=pairwise_distances(X2)

flag = np.allclose(D,D2)


plt.figure()


model=MDS(n_components=2,dissimilarity='precomputed',random_state=1)
out=model.fit_transform(D)
plt.scatter(out[:,0],out[:,1],**colorize)
plt.axis('equal')

