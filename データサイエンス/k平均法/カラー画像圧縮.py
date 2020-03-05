# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 10:10:37 2019

@author: ShimaLab
"""


import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import make_moons
from sklearn.cluster import SpectralClustering
from sklearn.datasets import load_digits
from scipy.stats import mode
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE
from sklearn.datasets import load_sample_image
from sklearn.cluster import MiniBatchKMeans

china=load_sample_image("china.jpg")
ax=plt.axes(xticks=[],yticks=[])
ax.imshow(china)

data=china/255
data=data.reshape(427*640,3)

def plot_pixel(data,title,colors=None,N=10000):
    if colors is None:
        colors=data
    
    rng=np.random.RandomState(0)
    i = rng.permutation(data.shape[0])[:N]
    colors=colors[i]
    R,G,B=data[i].T
    
    fig,ax=plt.subplots(1,2,figsize=(10,6))
    ax[0].scatter(R,G,color=colors,marker='.')
    ax[0].set(xlabel='Red',ylabel='Green',xlim=(0,1),ylim=(0,1))
    
    ax[1].scatter(R,B,color=colors,marker='.')
    ax[1].set(xlabel='Red',ylabel='Blue',xlim=(0,1),ylim=(0,1))
    
    fig.suptitle(title,size=20)


plot_pixel(data,title='Input color space:16 million possible colors')


plt.figure()

kmeans=MiniBatchKMeans(16)
kmeans.fit(data)
newcolors=kmeans.cluster_centers_[kmeans.predict(data)]
plot_pixel(data,colors=newcolors,title="Reduced color space:16 colors")


plt.figure()
china_recolored=newcolors.reshape(china.shape)
fig,ax=plt.subplots(1,2,figsize=(16,6),subplot_kw=dict(xticks=[],yticks=[]))
fig.subplots_adjust(wspace=0.05)
ax[0].imshow(china)
ax[0].set_title('Original Image',size=16)
ax[1].imshow(china_recolored)
ax[1].set_title('16-color Image',size=16)









