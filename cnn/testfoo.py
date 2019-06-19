#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 23:56:44 2017

@author: yangsong
"""

from mnist_utils import read_mnist
from mnist_utils import plot_image
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from keras.models import load_model

from sklearn import metrics
from sklearn.externals import joblib
import itertools



print('reading data...', end='')
data = np.load('nondigits.npy')


#X, y, classes = read_mnist('nondigits.npy')
print('done!')
print(data)

#plt.imshow(data, cmap='gray')
#plt.show()

plot_image(data, 0)
plot_image(data, 1)
plot_image(data, 2)
plot_image(data, 3)
#print(X)  # X is the image points
#print(y)  # y is the answers 
#print(classes)
#print('done!')

data = data.tolist()
print(type(data))
print(len(data))
newarray = []
for i in range(0,len(data)):
    partarray = []
    for j in range(0,len(data[0])):
        print(data[i][j])
        print(len(data[i][j]))
        partarray.extend(data[i][j])
    newarray.append(partarray)
    
print(newarray[0])
print(newarray[1])
print(newarray[2])
print(newarray[3])
    
newdata = np.array(newarray)
#newdata  /= 255
print(newdata.shape)


print()
#Z = np.load(file)
m, n = newdata.shape
X = np.float64(newdata[:, 0:n])
print(X)
#y = Z[:, n-1]
#classes = np.unique(y)

model2 = load_model('cnn_mnist_99.hdf5')
y_pred = model2.predict(X)
#y_pred = np.argmax(y_pred, axis=1)

