# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:26:01 2019

@author: ShimaLab
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import stats

from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
 
model = SVC(kernel='linear',C=1E10)
model.fit(X,y)

def plot_svc_decision_function(model,ax=None,plot_support=True):
    if ax is None:
        ax=plt.gca()
    
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
    x=np.linspace(xlim[0],xlim[1],30)
    y=np.linspace(ylim[0],ylim[1],30)
    
    Y,X=np.meshgrid(y,x)
    
    xy=np.vstack([X.ravel(),Y.ravel()]).T
    P=model.decision_function(xy).reshape(X.shape)
    
    ax.contour(X,Y,P,colors='k',levels=[-1,0,1],alpha=0.5,linestyles=['--','-','--'])
    
    if plot_support:
        ax.scatter(model.support_vectors_[:,0],model.support_vectors_[:,1],s=300,linewidth=1,
                   facecolors='none',edgecolors='black')
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    
    
    
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plot_svc_decision_function(model)









