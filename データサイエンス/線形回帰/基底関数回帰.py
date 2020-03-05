# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 18:49:45 2019

@author: ShimaLab
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

x = np.array([2,3,4])
poly = PolynomialFeatures(3,include_bias=False)
data = poly.fit_transform(x[:,None])

poly_model = make_pipeline(PolynomialFeatures(7),LinearRegression())

rng = np.random.RandomState(1)
x=10*rng.rand(50)
y=np.sin(x)+0.1*rng.randn(50)

xfit = np.linspace(0,10,1000)

poly_model.fit(x[:,np.newaxis],y)
yfit=poly_model.predict(xfit[:,np.newaxis])

plt.scatter(x,y)
plt.plot(xfit,yfit)