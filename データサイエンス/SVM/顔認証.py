# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:05:26 2019

@author: ShimaLab
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import stats

from sklearn.datasets.samples_generator import make_blobs
from sklearn.svm import SVC
from sklearn.datasets.samples_generator import make_circles
from mpl_toolkits import mplot3d
from ipywidgets import interact,fixed
from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


faces=fetch_lfw_people(min_faces_per_person=60)
print(faces.target_names)
print(faces.images.shape)

fig,ax=plt.subplots(3,5)
for i,axi in enumerate(ax.flat):
    axi.imshow(faces.images[i],cmap='bone')
    axi.set(xticks=[],yticks=[],xlabel=faces.target_names[faces.target[i]])

pca=PCA(n_components=150,whiten=True,random_state=42,svd_solver='randomized')
svc=SVC(kernel='rbf',class_weight='balanced')

model=make_pipeline(pca,svc)

X_train,X_test,y_train,y_test=train_test_split(faces.data,faces.target,random_state=42)

param_grid={'svc__C':[1,5,10,50],'svc__gamma':[0.00001,0.00005,0.001,0.005]}
grid=GridSearchCV(model,param_grid)

grid.fit(X_train,y_train)
print(grid.best_params_)

model=grid.best_estimator_
yfit=model.predict(X_test)


fig,ax=plt.subplots(4,6)
for i,axi in enumerate(ax.flat):
    axi.imshow(X_test[i].reshape(62,47),cmap='bone')
    axi.set(xticks=[],yticks=[])
    axi.set_ylabel(faces.target_names[yfit[i]].split()[-1],color='black'if yfit[i]==y_test[i] else 'red')

fig.suptitle('Predict Names; Incorrect Labels in Red',size=14)
    

print(classification_report(y_test,yfit,target_names=faces.target_names))
    
mat=confusion_matrix(y_test,yfit)
sns.heatmap(mat.T,square=True,annot=True,fmt='d',cbar=False,cmap='RdPu',xticklabels=faces.target_names,yticklabels=faces.target_names)
plt.xlabel('true label')
plt.ylabel('predicted label')










    
    
    
