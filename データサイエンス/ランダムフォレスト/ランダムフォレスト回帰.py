# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:49:31 2019

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
from sklearn.ensemble import RandomForestRegressor

rng=np.random.RandomState(42)

x=10*rng.rand(200)

def model(x,sigma=0.3):
    fast_oscillation=np.sin(5*x)
    slow_oscillation=np.sin(0.5*x)
    noise=sigma*rng.randn(len(x))
    
    return slow_oscillation+fast_oscillation+noise


y=model(x)
plt.errorbar(x,y,0.3,fmt='o')

forest=RandomForestRegressor(200)
forest.fit(x[:,np.newaxis],y)


xfit=np.linspace(0,10,1000)
yfit=forest.predict(xfit[:,None])
ytrue=model(xfit,sigma=0)

plt.errorbar(x,y,0.3,fmt='o',alpha=0.5)
plt.plot(xfit,yfit,'-r')
plt.plot(xfit,ytrue,'-k',alpha=0.5)
