# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 12:19:55 2019

@author: antho
"""

import pickle
import os
import numpy as np
import pandas as pd

def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']

        return X, Y

def load_CIFAR10(ROOT):
    """ load all of cifar """
    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_CIFAR_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte



# Invoke the above function to get our data.
X_train, y_train, X_test, y_test = load_CIFAR10('cifar-10-batches-py/')

#Convert our dataset into pandas.DataFrame so it can be concatenate
X_train = pd.DataFrame(X_train)
y_train = pd.DataFrame(y_train)
X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)

#Concatenate the images and labels
dataset_train = pd.concat([X_train,y_train], axis = 1)
dataset_test = pd.concat([X_test,y_test], axis = 1)

#Save our dataset to csv. csv file will be visible in your working directory
dataset_train.to_csv('dataset_train.csv', index = False)
dataset_test.to_csv('dataset_test.csv', index = False)