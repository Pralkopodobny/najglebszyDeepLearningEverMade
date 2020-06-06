# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
from gaborFilter import gabor_filter


def create_model():
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(512, activation='relu', kernel_initializer=keras.initializers.GlorotUniform(seed=None),
                           kernel_regularizer=keras.regularizers.l2()),
        keras.layers.Dense(256, activation='relu', bias_initializer=initializers.Constant(0.1),
                           kernel_initializer=keras.initializers.GlorotUniform(seed=None),
                           kernel_regularizer=keras.regularizers.l2()),
        keras.layers.Dense(10)
    ])

    model.compile(optimizer='SGD',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(x_train.shape)

x_train = x_train.reshape((x_train.shape[0], 28, 28))
x_test = x_test.reshape((x_test.shape[0], 28, 28))


model = create_model()
x_train_gabor = gabor_filter(x_train)
x_test_gabor = gabor_filter(x_test)
x_train_gabor, x_test_gabor = x_train_gabor.astype('float32') / 255, x_test_gabor.astype('float32') / 255

print(len(x_test_gabor))

model.fit(x_train_gabor,  y_train, validation_split=0.1, epochs=10)

test_loss, test_acc = model.evaluate(x_test_gabor, y_test, verbose=2)

print('\nTest accuracy:', test_acc)
model.summary()



