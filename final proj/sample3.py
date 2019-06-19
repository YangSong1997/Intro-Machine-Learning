#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 16:53:23 2017

@author: yangsong
"""

import pandas as pd
import numpy  as np


# A function that computes the errors.
from math import sqrt
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

y_train = df_train.iloc[:,1:3]
y_test  = df_test.iloc[:,1:3]



y_cols = ['dem', 'gop']

x_cols = [   'total_votes_2012', 'Romney','votes_dem_2012','PST045214', 'Obama','LND110210', 'POP060210',
     'PST040210', 'PST120214','POP010210','RHI125214','NES010213','SEX255214','HSG096213','WTN220207']

x_train = x_train.loc[:,x_cols].as_matrix().astype(float)
y_train = y_train.loc[:,y_cols].as_matrix().astype(float)

x_test = x_test.loc[:,x_cols].as_matrix().astype(float)
y_test = y_test.loc[:,y_cols].as_matrix().astype(float)

  

y_train = {'dem': y_train[:,0], 
           'gop': y_train[:,1]}
y_test  = {'dem': y_test[:,0], 
           'gop': y_test[:,1]}


from sklearn.linear_model import LinearRegression, Lasso

clf={}
for party in y_train:
  clf[party] = Lasso(alpha=0.9, fit_intercept=True, normalize=True, copy_X=True, tol=0.001, random_state=2)
  print('training Lasso classifier...'.format(party), end='')
  clf[party].fit(x_train, y_train[party])
  print('done!')



# We copy the predictors we have built to an iterable object.
pred = {}
pred['dem'] = clf['dem'].predict(x_test)
pred['gop'] = clf['gop'].predict(x_test)
y_pred = check_results(clf, x_test, y_test)


print('{0:16s}   {1:14s}  {2:14s}'.format('', 'dem', 'gop'))
for k in range(len(x_cols)):
    print('{0:16s}  {1: e}  {2: e}'.format(x_cols[k], clf['dem'].coef_[k], clf['gop'].coef_[k]))



