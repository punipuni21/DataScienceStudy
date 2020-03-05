# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 13:51:03 2019

@author: ShimaLab
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import stats

from sklearn.datasets.samples_generator import make_blobs

#2クラス分類
X,y = make_blobs(n_samples=50,centers = 2,random_state=0,cluster_std=0.60)
#plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')


xfit=np.linspace(-1,3.5)
plt.scatter(X[:,0],X[:,1],c=y,s=50,cmap='autumn')
plt.plot([0.6],[2.1],'x',color='red',markeredgewidth=2,markersize=10)

for m,b in [(1,0.65),(0.5,1.6),(-0.2,2.9)]:
    plt.plot(xfit,m*xfit+b)

plt.xlim(-1,3.5)

