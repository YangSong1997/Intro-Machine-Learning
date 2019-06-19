#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 10:38:05 2017

@author: yangsong
"""

import numpy as np
from keras import utils as kutils
from keras import backend as K
import tensorflow as tf
from keras import models, layers, losses, optimizers
from IPython.display import Image
from sklearn import metrics
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator


from mnist_utils import read_mnist
from mnist_utils import plot_image
from matplotlib import pyplot as plt
from PIL import Image

from sklearn import metrics
from sklearn.externals import joblib
import itertools


def read_mnist (vile):
    """
    This function reads the MNIST .npy files and returns the feature vectors and their associated
    class labels, and a list of the class labels.
    """
    Z = np.load(vile)
    m, n = Z.shape
    X = np.float32(Z[:, 0:n-1])
    y = Z[:, n-1]
    classes = np.unique(y)
    return X, y, classes

def compute_metrics (classifier, X_test, y_test, classes):
    """
    This function computes and prints various performance measures for a classifier.
    """
    # Use the classifier to make predictions for the test set.
    y_pred = classifier.predict(X_test)
    
    # Choose the class with the highest estimated probability.
    y_pred = np.argmax(y_pred, axis=1)

    print('Classes:', classes, '\n')

    # Compute the confusion matrix.
    cm = metrics.confusion_matrix(y_test, y_pred, labels=classes)
    print('Confusion matrix, without normalization')
    print(cm, '\n')

    # Normalize the confusion matrix by row (i.e by the number of samples in each class).
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    np.set_printoptions(precision=3, linewidth=132)
    print('Normalized confusion matrix')
    print(cm_normalized, '\n')

    # The confusion matrix as percentages.
    cm_percentage = 100 * cm_normalized
    print('Confusion matrix as percentages')
    print(np.array2string(cm_percentage, formatter={'float_kind':lambda x: "%6.2f" % x}), '\n')
    
    # Precision, recall, and f-score.
    print(metrics.classification_report(y_test, y_pred, digits=3))

    return cm



# First part ----------------------------------------------------------------------
print('Reading the data...', end='')
x_train, y_train, classes = read_mnist('mnist_train.npy')
x_test,  y_test,  classes = read_mnist('mnist_test.npy')
print('done!')

print('Shape of x_train after prefrobincation:', x_train.shape)
print('Number of training cases: {0:5d}'.format(x_train.shape[0]))
print('Number of test cases:     {0:5d}'.format(x_test.shape[0]))


# Second Part ----------------------------------------------------------------------
print('Scaling data to the range [0,1]...', end='')
x_train /= 255
x_test  /= 255
print('done!')


# Third Part ----------------------------------------------------------------------
# The number of classes.
num_classes = 10

print('Encoding the class labels...', end='')
y_train_enc = kutils.to_categorical(y_train, num_classes)
y_test_enc  = kutils.to_categorical(y_test,  num_classes)
print('done!')



# Perform some pre-flight checks -------------------------------------------------
# The input image dimensions.
img_rows, img_cols = 28, 28

# The number of classes.
num_classes = 10

# Check whether the channels (e.g., RGB) should come first or second and indicate this in
# the tuple input_shape.  This isn't an issue for the grayscale MNIST data.
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test  = x_test.reshape(x_test.shape[0],   1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test  = x_test.reshape(x_test.shape[0],   img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

print('Shape of x_train after prefrobincation:', x_train.shape)
print('Number of training cases: {0:5d}'.format(x_train.shape[0]))
print('Number of test cases:     {0:5d}'.format(x_test.shape[0]))





# Specify the CNN -----------------------------------------------------------------
print('constructing the model...', end='')

model = models.Sequential()
model.add(layers.Conv2D(64, kernel_size=(3, 3),
          activation='relu',
          input_shape=input_shape))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
#model.add(layers.Dropout(0.25))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='softmax'))

model.compile(loss=losses.categorical_crossentropy,
              optimizer=optimizers.Adadelta(),
              metrics=['accuracy'])

print('done!')


model.summary()

for layer in model.layers:
  print(layer)

#kutils.plot_model(model, to_file='model.png')
#Image('model.png')


batch_size = 128
epochs = 1

model.fit(x_train, y_train_enc,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test_enc))

score = model.evaluate(x_test, y_test_enc, verbose=0)
print('Test loss:    ', score[0])
print('Test accuracy:', score[1])


# Softmax ---------------------------------------------------------------------
get_softmax_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                      [model.layers[5].output])

num_cases = 11
softmax = get_softmax_layer_output([x_test[0:num_cases,:], 0])[0]

print('label  estimated probabilities by class')
print('       ', end='')
for c in range(0, num_classes):
    print('{0:3d}  '.format(c), end='')
print()
for t in range(0, num_cases):
    print('{0:<5d}  '.format(y_test[t]), end='')
    for c in range(0, num_classes):
        print('{0:.1f}  '.format(softmax[t][c]), end='')
    print()
#    
#H = np.load('hillary.npy')
#H = H.reshape(H.shape[0], img_rows, img_cols, 1)
#
#
#get_softmax_layer_output([H, 0])[0]


# Compute metrics
compute_metrics(model, x_test, y_test, classes)




#batch_size = 128
#epochs = 2
#
#model.fit(x_train, y_train,
#          batch_size=batch_size,
#          epochs=epochs,
#          verbose=1,
#          validation_data=(x_test, y_test),
#          initial_epoch = 1)
#
#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:    ', score[0])
#print('Test accuracy:', score[1])


print(model.layers[0].get_weights())
print(model.layers[1].get_weights())
print(model.layers[2].get_weights())
print(model.layers[3].get_weights())
print(model.layers[4].get_weights())




print('saving the model...', end='')
model.save('cnn_mnist_best.hdf5')
print('done!')





# --------------------------------------------------------------------------------

#model2 = load_model('cnn_mnist.hdf5')
#del model2

#print('reconstructing the model...', end='')
#
#aug = models.Sequential()
#aug.add(layers.Conv2D(32, kernel_size=(3, 3),
#           activation='relu',
#           input_shape=input_shape))
#aug.add(layers.Conv2D(64, (3, 3), activation='relu'))
#aug.add(layers.MaxPooling2D(pool_size=(2, 2)))
#aug.add(layers.Dropout(0.25))
#aug.add(layers.Flatten())
#aug.add(layers.Dense(128, activation='relu'))
#aug.add(layers.Dropout(0.5))
#aug.add(layers.Dense(num_classes, activation='softmax'))
#
#aug.compile(loss=losses.categorical_crossentropy,
#              optimizer=optimizers.Adadelta(),
#              metrics=['accuracy'])
#
#print('done!')





#
#batch_size = 128
#num_classes = 10
#epochs = 1
#
#datagen = ImageDataGenerator(
#  width_shift_range=0.1,               # randomly shift images horizontally (fraction of total width)
#  height_shift_range=0.1,              # randomly shift images vertically (fraction of total height)
#                                       # The remaining options are here for illustrative purposes only.
#  featurewise_center=False,            # set input mean to 0 over the dataset
#  samplewise_center=False,             # set each sample mean to 0
#  featurewise_std_normalization=False, # divide inputs by std of the dataset
#  samplewise_std_normalization=False,  # divide each input by its std
#  zca_whitening=False,                 # apply ZCA whitening 
#  rotation_range=0,                    # randomly rotate images in the range (in degrees, 0 to 180)
#  horizontal_flip=False,               # randomly flip images
#  vertical_flip=False)                 # randomly flip images
#
## Compute quantities required for feature-wise normalization
## (std, mean, and principal components if ZCA whitening is applied).
#datagen.fit(x_train)
#
## Fit the model on the batches generated by datagen.flow().
#model2.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
#                          steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
#                          epochs=epochs,
#                          validation_data=(x_test, y_test),
#                          workers=4)
#
#score = model2.evaluate(x_test, y_test, verbose=0)
#print('Test loss:    ', score[0])
#print('Test accuracy:', score[1])




# =============================================================================
# 
# print('reading data...', end='')
# data = np.load('nondigits.npy')
# 
# 
# #X, y, classes = read_mnist('nondigits.npy')
# print('done!')
# print(data)
# 
# #plt.imshow(data, cmap='gray')
# #plt.show()
# 
# plot_image(data, 0)
# plot_image(data, 1)
# plot_image(data, 2)
# plot_image(data, 3)
# #print(X)  # X is the image points
# #print(y)  # y is the answers 
# #print(classes)
# #print('done!')
# 
# data = data.tolist()
# print(type(data))
# print(len(data))
# newarray = []
# for i in range(0,len(data)):
#     partarray = []
#     for j in range(0,len(data[0])):
#         print(data[i][j])
#         print(len(data[i][j]))
#         partarray.extend(data[i][j])
#     newarray.append(partarray)
#     
# #print(newarray[0])
# #print(newarray[1])
# #print(newarray[2])
# #print(newarray[3])
#     
# newdata = np.array(newarray)
# 
# #model2 = load_model('cnn_mnist.hdf5')
# y_pred = model.predict(newdata)
# y_pred = np.argmax(y_pred, axis=1)
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# =============================================================================
