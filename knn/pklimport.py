#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 18:33:00 2017

@author: yangsong
"""

import pickle
import matplotlib.cm as cm
import matplotlib.pyplot as plt

with open('2.pkl', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)
    
train_x, train_y = train_set
plt.imshow(train_x[0].reshape((28, 28)), cmap=cm.Greys_r)
plt.show()