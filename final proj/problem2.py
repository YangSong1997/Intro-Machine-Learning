#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:36:26 2017

@author: yangsong
"""
from sklearn import linear_model

import pandas as pd
import numpy  as np

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from math import sqrt
from sklearn.preprocessing import MinMaxScaler



def check_results(clf, x_test, y_test):
  y_pred = r = relerr = {}
  for party in clf:
    y_pred[party] = clf[party].predict(x_test).reshape(y_test[party].shape)

    r = y_pred[party] - y_test[party]
    relerr = r / y_test[party]

    print('{0} classifier:'.format(party))
    print('  RMS    error: {0:>7.0f}'.format(sqrt(np.mean(r**2))))
    print('  Mean   error: {0:>7.0f}'.format(np.mean(np.abs(r))))
    print('  Median error: {0:>7.0f}'.format(np.median(np.abs(r))))
    print('  Number of significant (> 1.0e-4) model coefficients: {0}\n'.
              format((np.abs(clf[party].coef_) > 1.0e-4).sum()))
  return y_pred



print('reading data...', end='')
df_train = pd.read_csv('train_vote16.csv')
df_test  = pd.read_csv('test_vote16.csv')
print('done!')

x_train = df_train.iloc[:,3:]
x_test  = df_test.iloc[:,3:]

print("1",x_test.shape)


y_train = df_train.iloc[:,1:3]
y_test  = df_test.iloc[:,1:3]
#print(y_train)
#print(y_train['dem'])
#print(y_test)





y_train_dem = y_train.iloc[:,0]
y_train_gop = y_train.iloc[:,1]
y_test_dem = y_test.iloc[:,0]
y_test_gop = y_test.iloc[:,1]
#print(y_train_dem)
#print(y_train_gop)
print()


scaler = StandardScaler()

x_cols = ["total_votes_2012","votes_dem_2012","votes_gop_2012","Obama","Romney","PST045214","PST040210","PST120214","POP010210","AGE135214","AGE295214","AGE775214","SEX255214","RHI125214","RHI225214","RHI325214","RHI425214","RHI525214","RHI625214","RHI725214","RHI825214","POP715213","POP645213","POP815213","EDU635213","EDU685213","VET605213","LFE305213","HSG010214","HSG445213","HSG096213","HSG495213","HSD410213","HSD310213","INC910213","INC110213","PVY020213","BZA010213","BZA110213","BZA115213","NES010213","SBO001207","SBO315207","SBO115207","SBO215207","SBO515207","SBO415207","SBO015207","MAN450207","WTN220207","RTN130207","RTN131207","AFN120207","BPS030214","LND110210","POP060210"]

x_train_var = x_train.loc[:,x_cols].as_matrix().astype(float) # training X
x_test_var = x_test.loc[:,x_cols].as_matrix().astype(float)  # testing X

print("2",x_test_var.shape)


x_train_fit = scaler.fit_transform(x_train_var)
x_test_fit = scaler.fit_transform(x_test_var)


#print(type(x_train_var))
#print(x_train_var)

#scaler = StandardScaler()
#pca = PCA()

#x_train_fit = scaler.fit_transform(x_train_var)
#x_test_fit = scaler.fit_transform(x_test_var)


lasso = linear_model.Lasso(alpha=300.0,copy_X=True)

clf = {}
#pipe_lr = Pipeline([('scl', scl),
#            ('pca', pca),
#            ('lasso', lasso)])




#y_train_party = 
for col in y_train:
    y_train_party = y_train[col]
    print(col)
    print(y_train_party)
    
    lasso_party = lasso.fit(x_train_fit, y_train_party)
    clf[col] = lasso_party


#lasso_dem = lasso.fit(x_train_fit, y_train_dem)
#lasso_gop = lasso.fit(x_train_fit, y_train_gop)
#clf['dem'] = lasso_dem
#clf['gop'] = lasso_gop


print("3",x_test_fit.shape)
print("4",y_test.shape)


y_pred = check_results(clf, x_test_fit, y_test)

