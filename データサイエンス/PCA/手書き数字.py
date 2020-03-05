# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 09:29:03 2019

@author: ShimaLab
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()


from sklearn.decomposition import PCA
from sklearn.datasets import load_digits

digits=load_digits()


pca=PCA(n_components=2)
projected=pca.fit_transform(digits.data)
print(digits.data.shape)
print(projected.shape)

plt.scatter(projected[:,0],projected[:,1],c=digits.target,edgecolors='none',alpha=0.5,cmap=plt.cm.get_cmap('hsv',10))


plt.xlabel('component1')
plt.ylabel('component2')
plt.colorbar()

plt.figure()
pca=PCA().fit(digits.data)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of component')
plt.ylabel('cumulative explained variance')


