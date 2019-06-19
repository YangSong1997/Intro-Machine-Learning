#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 10 22:36:23 2017

@author: yangsong
"""
import numpy as np
from mnist_utils import compute_metrics
from mnist_utils import save_classifier
from mnist_utils import read_mnist
from mnist_utils import plot_image
#from sklearn import svm
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import time




print('reading data...', end='')
data = np.load('mnist_public.npy')
print(data.shape)
X, y, classes = read_mnist('mnist_public.npy')
print(X)  # X is the image points
print(y)  # y is the answers 
print(classes)
print('done!')
print(X.shape)
#for el in X:
 #   print(el)

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10,train_size=0.90,
                                                    random_state=0)

print(type(X_train))