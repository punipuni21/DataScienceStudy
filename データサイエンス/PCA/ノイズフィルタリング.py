# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 10:47:48 2019

@author: ShimaLab
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns;sns.set()

from sklearn.decomposition import PCA

def plot_digits(data):
    fig,axes=plt.subplots(4,10,figsize=(10,4),subplot_kw={'xticks':[],'yticks':[]},
                          gridspec_kw=dict(hspace=0.1,wspace=0.1))
    for i,ax in enumerate(axes.flat):
        ax.imshow(data[i].reshape(8,8),cmap='binary',interpolation='nearest',clim=(0,16))
    
    
#ノイズなし
plot_digits(digits.data)

#ノイズをのせる
plt.figure()
np.random.seed(42)
noisy=np.random.normal(digits.data,4)
plot_digits(noisy)

pca=PCA(0.50).fit(noisy)
print("components:",pca.n_components_)

components=pca.transform(noisy)
filtered=pca.inverse_transform(components)
plot_digits(filtered)