# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 14:40:12 2019

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
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix

digits=load_digits()
digits.keys()

fig=plt.figure(figsize=(6,6))
fig.subplots_adjust(left=0,right=1,bottom=0,top=1,hspace=0.05,wspace=0.05)

# =============================================================================
# for i in range(64):
#     ax=fig.add_subplot(8,8,i+1,xticks=[],yticks=[])
#     ax.imshow(digits.images[i],cmap=plt.cm.binary,interpolation='nearest')
#     ax.text(0,7,str(digits.target[i]))
# =============================================================================
    
Xtrain,Xtest,ytrain,ytest=train_test_split(digits.data,digits.target,random_state=0)

model=RandomForestClassifier(n_estimators=1000)
model.fit(Xtrain,ytrain)
ypred=model.predict(Xtest)

print(metrics.classification_report(ypred,ytest))

mat=confusion_matrix(ytest,ypred)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,cmap='RdPu')

plt.xlabel('true label')
plt.ylabel('predicted label')