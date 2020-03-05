# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:33:16 2019

@author: ShimaLab
"""


import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from mpl_toolkits import mplot3d
from sklearn.manifold import LocallyLinearEmbedding



def make_hello_s_curve(X):
    t=(X[:,0]-2)*0.75*np.pi
    x=np.sin(t)
    y=X[:,1]
    z=np.sign(t)*(np.cos(t)-1)
    return np.vstack((x,y,z)).T

XS=make_hello_s_curve(X)


ax=plt.axes(projection='3d')
ax.scatter3D(XS[:,0],XS[:,1],XS[:,2],**colorize)

plt.figure()

model=MDS(n_components=2,random_state=2)
outS=model.fit_transform(XS)
plt.scatter(outS[:,0],outS[:,1],**colorize)
plt.axis('equal')


plt.figure()


model=LocallyLinearEmbedding(n_neighbors=100,n_components=2,method='modified',eigen_solver='dense')

out=model.fit_transform(XS)

fig,ax=plt.subplots()
ax.scatter(out[:,0],out[:,1],**colorize)
ax.set_ylim(0.15,-0.15)