# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 14:22:58 2019

@author: ShimaLab
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

x=np.array([1,2,3,4,5,])
y=np.array([4,2,1,3,7])
#plt.scatter(x,y)

X=x[:,np.newaxis]

# =============================================================================
# model = LinearRegression().fit(X,y)
# yfit=model.predict(X)
# plt.scatter(x,y)
# plt.plot(x,yfit)
# =============================================================================

poly = PolynomialFeatures(degree=3,include_bias=False)
X2=poly.fit_transform(X)


model=LinearRegression().fit(X2,y)
yfit=model.predict(X2)
plt.scatter(x,y)
plt.plot(x,yfit)