#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 10:32:23 2017

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

print(X_test)
print(X_test.shape)


X_train_fit = scaler.fit_transform(X_train)
X_test_fit = scaler.fit_transform(X_test)

start = time.time()
svm = SVC(kernel='rbf', C=100.0, gamma=0.01,random_state=0)
Svc = svm.fit(X_train_fit,y_train)
end = time.time()
print(end - start)
compute_metrics(Svc, X_test_fit, y_test, classes)
save_classifier(Svc,'svc.pkl')


lr = LogisticRegression(C=100.0, random_state=0)
log_reg = lr.fit(X_train_fit, y_train)
compute_metrics(log_reg, X_test_fit, y_test, classes)
save_classifier(log_reg,'lgr.pkl')


tree = DecisionTreeClassifier(criterion='entropy', max_depth=7, random_state=0)
dec_tree = tree.fit(X_train_fit, y_train)
compute_metrics(dec_tree, X_test_fit, y_test, classes)
save_classifier(dec_tree,'tree.pkl')



