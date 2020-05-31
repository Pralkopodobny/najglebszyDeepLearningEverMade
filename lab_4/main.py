import utils.mnist_reader as mnist_reader
import pickle
import warnings
from time import sleep

import matplotlib.pyplot as plt
import numpy as np

from K_NN import model_selection_knn

X_train, y_train = mnist_reader.load_mnist('data/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('data/fashion', kind='t10k')
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
k_values = [1,2,3]
error_best, best_k, errors = model_selection_knn(X_test, X_train, y_test, y_train, k_values)
print (error_best)
print (best_k)
print (errors)