import numpy as np
import matplotlib.pyplot as plt
import os
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, adam
from keras.utils import np_utils

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

np.random.seed(100)

classes = 10
epochs = 100
batch_size = 128

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train = (x_train - np.mean(x_train)) / np.std(x_train)  # Gaussian Normalization
x_test = (x_test - np.mean(x_test)) / np.std(x_test)

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

y_train = np_utils.to_categorical(y_train, classes)
y_test = np_utils.to_categorical(y_test, classes)

model = Sequential()
model.add(Dense(512, input_shape=(784,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(classes))
model.add(Activation("softmax"))

optimiser = adam()

model.compile(loss="categorical_crossentropy", optimizer=optimiser, metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2)

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=0)
print('\nTest accuracy:', test_acc)
