import utils.mnist_reader as mnist_reader
import pickle
import warnings
from time import sleep

import matplotlib.pyplot as plt
import numpy as np

from K_NN import model_selection_knn

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
X_train, X_test = X_train.astype('float32') / 255, X_test.astype('float32') / 255
"""
threshold = 100
for i, image in enumerate(X_train):
    image = image > threshold

for i, image in enumerate(X_test):
    image = image > threshold

"""
kek_train = X_train[0:1000,:]
kek_val = X_test[1000:1100,:]
kek_train_labels = y_train[0:1000]
kek_val_labels = y_test[1000:1100]


print("zabawa")
k_values = [3]
error_best, best_k, errors = model_selection_knn(kek_val, kek_train, kek_val_labels, kek_train_labels, k_values)
print (error_best)
print (best_k)
print (errors)
