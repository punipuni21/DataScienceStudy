# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:02:59 2019

@author: ShimaLab
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.datasets import make_blobs
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestClassifier


X,y=make_blobs(n_samples=300,centers=4,random_state=0,cluster_std=1.0)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='rainbow')    

tree=DecisionTreeClassifier().fit(X,y)

def visualisze_classifier(model,X,y,ax=None,cmap='rainbow'):
    ax=ax or plt.gca()
    
#    学習データのプロット
    ax.scatter(X[:,0],X[:,1],c=y,s=30,cmap=cmap,clim=(y.min(),y.max()),zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim=ax.get_xlim()
    ylim=ax.get_ylim()
    
#   推定器への当てはめ    
    model.fit(X,y)
    xx,yy=np.meshgrid(np.linspace(*xlim,num=200),np.linspace(*ylim,num=200))
    Z=model.predict(np.c_[xx.ravel(),yy.ravel()]).reshape(xx.shape)
    
#    色に分けて結果を表示
    n_class=len(np.unique(y))
    contours=ax.contourf(xx,yy,Z,alpha=0.3,levels=np.arange(n_class+1)-0.5,cmap=cmap,clim=(y.min(),y.max()),zorder=1)
    ax.set(xlim=xlim,ylim=ylim)



#決定木分類器の可視化
visualisze_classifier(DecisionTreeClassifier(),X,y)



#ランダム決定木のアンサンブルによる協会検出
#ここではデータを80%の大きさでランダムに分割し，推定木に当てはめる
tree=DecisionTreeClassifier()
bag=BaggingRegressor(tree,n_estimators=100,max_samples=0.8,random_state=1)

bag.fit(X,y)
visualisze_classifier(bag,X,y)


#決定木の最適化されたアンサンブルであるランダムフォレストによる境界検知


















