# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 08:50:57 2019

@author: ShimaLab
"""

import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import numpy as np


def make_hello(N=1000,rseed=42):
    fig ,ax=plt.subplots(figsize=(4,1))
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    ax.axis('off')
    ax.text(0.5,0.4,'HELLO',va='center',ha='center',weight='bold',size=85)
    fig.savefig('hello.png')
    plt.close()
    
    
    from matplotlib.image import imread
    data=imread('hello.png')[::-1,:,0].T
    rng=np.random.RandomState(rseed)
    X=rng.rand(4*N,2)
    i,j=(X*data.shape).astype(int).T
    mask=(data[i,j]<1)
    X=X[mask]
    X[:,0]*=(data.shape[0]/data.shape[1])
    X=X[:N]
    return X[np.argsort(X[:,0])]


X=make_hello(1000)
colorize=dict(c=X[:,0],cmap=plt.cm.get_cmap('rainbow',5))
plt.scatter(X[:,0],X[:,1],**colorize)
plt.axis('equal')

