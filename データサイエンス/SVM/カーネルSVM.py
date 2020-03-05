# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 20:08:10 2019

@author: ShimaLab
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import stats

from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_circles
from mpl_toolkits import mplot3d
from ipywidgets import interact,fixed


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
    
    
def plot_3D(elev=30,azim=30,X=X,y=y):
    ax=plt.subplot(projection='3d')
    ax.scatter3D(X[:,0],X[:,1],r,c=y,s=50,cmap='autumn')
    ax.view_init(elev=elev,azim=azim)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    
#これでは分離できない
X,y=make_circles(100,factor=.1,noise=.1)

clf=SVC(kernel='linear').fit(X,y)


# =============================================================================
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plot_svc_decision_function(clf,plot_support=False)
# =============================================================================


#放射基底関数を利用する
r=np.exp(-(X**2).sum(1))

interact(plot_3D,elev=[30,60],azip=(-180,180),X=fixed(X),y=fixed(y))


#カーネル化したSVM
clf=SVC(kernel='rbf',C=1E6).fit(X,y)
clf.fit(X,y)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plot_svc_decision_function(clf)
plt.scatter(clf.support_vectors_[:,0],clf.support_vectors_[:,1],s=300,lw=1,facecolors='none')





