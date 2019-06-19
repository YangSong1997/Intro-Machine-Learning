#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:36:26 2017

@author: yangsong
"""
from sklearn import linear_model

import pandas as pd
import numpy  as np
from sklearn import metrics

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.preprocessing import MinMaxScaler


print('reading data...', end='')
df_train = pd.read_csv('train_vote16.csv')
df_test  = pd.read_csv('test_vote16.csv')
print('done!')

x_train = df_train.iloc[:,3:]
x_test  = df_test.iloc[:,3:]

y_train = df_train.iloc[:,0]
y_test  = df_test.iloc[:,0]

scaler = StandardScaler()
pca = PCA()

x_cols = ["total_votes_2012","votes_dem_2012","votes_gop_2012","Obama","Romney","PST045214","PST040210","PST120214","POP010210","AGE135214","AGE295214","AGE775214","SEX255214","RHI125214","RHI225214","RHI325214","RHI425214","RHI525214","RHI625214","RHI725214","RHI825214","POP715213","POP645213","POP815213","EDU635213","EDU685213","VET605213","LFE305213","HSG010214","HSG445213","HSG096213","HSG495213","HSD410213","HSD310213","INC910213","INC110213","PVY020213","BZA010213","BZA110213","BZA115213","NES010213","SBO001207","SBO315207","SBO115207","SBO215207","SBO515207","SBO415207","SBO015207","MAN450207","WTN220207","RTN130207","RTN131207","AFN120207","BPS030214","LND110210","POP060210"]

x_train_var = x_train.loc[:,x_cols].as_matrix().astype(float) # training X
x_test_var = x_test.loc[:,x_cols].as_matrix().astype(float)  # testing X

print("2",x_test_var.shape)


#x_train_fit = scaler.fit_transform(x_train_var)
#x_test_fit = scaler.fit_transform(x_test_var)





#pipe_lr = Pipeline([('scl', scl),
#            ('pca', pca),
#            ('lasso', lasso)])

svm = SVC(kernel='rbf', C=100.0, gamma=0.01,random_state=0)
#Svc = svm.fit(x_train_fit,y_train)
# 0.982

#lr = LogisticRegression(C=100.0, random_state=0)
#log_reg = lr.fit(x_train_fit, y_train)
# 0.980

#tree = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=0)
#dec_tree = tree.fit(x_train_fit, y_train)
# 0.924

#knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
#KNN = knn.fit(x_train_fit, y_train)
# 0.955

pipe_lr = Pipeline([('scaler', scaler),
            ('pca', pca),
            ('tree', tree)])

clf = pipe_lr.fit(x_train_var, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(x_test_var, y_test))

#y_pred = KNN.predict(x_test_fit)
#print(metrics.accuracy_score(y_test,y_pred))

