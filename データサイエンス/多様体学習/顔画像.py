# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:00:16 2019

@author: ShimaLab
"""


import matplotlib.pyplot as plt
import seaborn as sns;sns.set()
import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.manifold import MDS
from mpl_toolkits import mplot3d
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from matplotlib import offsetbox

faces=fetch_lfw_people(min_faces_per_person=30)

fig,ax=plt.subplots(4,8,subplot_kw=dict(xticks=[],yticks=[]))
for i,axi in enumerate(ax.flat):
    axi.imshow(faces.images[i],cmap='gray')
    

plt.figure()
model=PCA(100,svd_solver='randomized',).fit(faces.data)
plt.plot(np.cumsum(model.explained_variance_ratio_))
plt.xlabel('n components')
plt.ylabel('cummulative variance')


# =============================================================================
# model=Isomap(n_components=2)
# proj=model.fit_transform(faces.data)
# =============================================================================

def plot_components(data,model,images=None,ax=None,thumb_frac=0.05,cmap='gray'):
    ax=ax or plt.gca()
    
    proj=model.fit_transform(data)
    ax.plot(proj[:,0],proj[:,1],'.k')
    
    if images is not None:
        min_dist_2=(thumb_frac*max(proj.max(0)-proj.min(0)))**2
        shown_images=np.array([2*proj.max(0)])
        for i in range(data.shape[0]):
            dist=np.sum((proj[i]-shown_images)**2,1)
            if np.min(dist) < min_dist_2:
                continue
            shown_images=np.vstack([shown_images,proj[i]])
            imagebox=offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i],cmap=cmap),proj[i])
            ax.add_artist(imagebox)
            
            
fig,ax=plt.subplots(figsize=(10,10))
plot_components(faces.data,model=Isomap(n_components=2),images=faces.images[:,::2,::2])















