# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:24:53 2019

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

from sklearn.datasets import fetch_mldata

mnist=fetch_mldata('Mnist original')

fig,ax=plt.subplots(6,8,subplot_kw=dict(xticks=[],yticks=[]))
for i,axi in enumerate(ax.flat):
    axi.imshow(mnist.data[1250*i].reshape(28,28),cmap='gray_r')
    
    
    
    