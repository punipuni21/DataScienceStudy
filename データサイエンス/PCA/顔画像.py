# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:03:07 2019

@author: ShimaLab
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA


faces=fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

pca=PCA(80,svd_solver='randomized')
pca.fit(faces.data)


fig,axes=plt.subplots(3,8,figsize=(9,4),subplot_kw={'xticks':[],'yticks':[]},gridspec_kw=dict(hspace=0.1,wspace=0.1))

for i,ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62,47),cmap='bone')


plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of component')
plt.ylabel('cumulative explained variance')


plt.figure()
pca=PCA(80,svd_solver='randomized').fit(faces.data)
components=pca.transform(faces.data)
projected=pca.inverse_transform(components)
fig,ax=plt.subplots(2,10,figsize=(10,2.5),subplot_kw={'xticks':[],'yticks':[]},gridspec_kw=dict(hspace=0.1,wspace=0.1))

for i in range(10):
    ax[0,i].imshow(faces.data[i].reshape(62,47),cmap='binary_r')
    ax[1,i].imshow(projected[i].reshape(62,47),cmap='binary_r')

ax[0,0].set_ylabel('full-dim\ninput')
ax[1,0].set_ylabel('80-dim\ninput')