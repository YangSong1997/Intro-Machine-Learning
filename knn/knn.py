#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 13:53:03 2017

@author: yangsong
"""

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

from sklearn.neighbors import KNeighborsClassifier


print('reading data...', end='')
X, y, classes = read_mnist('mnist_public.npy')
print(X)  # X is the image points
print(y)  # y is the answers 
print('done!')

scaler = MinMaxScaler()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=2000,train_size=40000,
                                                    random_state=0)


X_train_fit = scaler.fit_transform(X_train)
X_test_fit = scaler.fit_transform(X_test)



knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
KNN = knn.fit(X_train_fit, y_train)


#svm = SVC(kernel='rbf', C=100.0, gamma=0.01,random_state=0)
#Svc = svm.fit(X_train_fit,y_train)
compute_metrics(KNN, X_test_fit, y_test, classes)
save_classifier(KNN,'1.pkl')



