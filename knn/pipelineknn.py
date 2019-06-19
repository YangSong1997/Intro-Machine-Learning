#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 11:00:41 2017

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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

import time

print('reading data...', end='')
X, y, classes = read_mnist('mnist_public.npy')
print(X)  # X is the image points
print(y)  # y is the answers 
print('done!')


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,train_size=0.9,
                                                    random_state=5)


scl = MinMaxScaler()
pca = PCA()
knn = KNeighborsClassifier(n_neighbors=3, weights = 'uniform',algorithm = 'brute', leaf_size = 50, p=2, metric='minkowski',metric_params = None, n_jobs = -1)


start = time.time()
pipe_lr = Pipeline([('scl', scl),
            ('pca', pca),
            ('knn', knn)])

knnclassifier = pipe_lr.fit(X_train, y_train)
print('Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test))
end = time.time()
print(end - start)
y_pred = pipe_lr.predict(X_test)
print(y_pred)

compute_metrics(knnclassifier, X_test, y_test, classes)
save_classifier(knnclassifier,'knn.pkl')





