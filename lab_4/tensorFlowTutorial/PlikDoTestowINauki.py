import numpy as np
import os
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import adam
from keras.utils import np_utils

# STEP 1.  DOWNLOAD THE DATA
np.random.seed(100)
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
# STEP 2. PROCESS THE DATA
X_train = X_train.reshape(50000, 3072)
X_test = X_test.reshape(10000, 3072)

X_train = (X_train - np.mean(X_train)) / np.std(X_train)  # Gaussian Normalization
X_test = (X_test - np.mean(X_test)) / np.std(X_test)  # Gaussian Normalization

labels = 10
y_train = np_utils.to_categorical(y_train, labels)
y_test = np_utils.to_categorical(y_test, labels)

model = Sequential()
model.add(Dense(512, input_shape=(3072,)))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(120))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(labels))
model.add(Activation("sigmoid"))

adam = adam(0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

model.fit(X_train,y_train, batch_size=1000, nb_epoch=10, validation_data=(X_test, y_test))

score = model.evaluate(X_test, y_test, verbose=0)
print('Test acc: ', score[1])

model.predict_classes(X_test)

