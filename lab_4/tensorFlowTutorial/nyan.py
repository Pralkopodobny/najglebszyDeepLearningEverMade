# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import initializers

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.layers import Activation, Dense, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.optimizers import adam
from tensorflow.python.keras.utils.vis_utils import plot_model

print(tf.__version__)
fashion_mnist = keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(x_train.shape)

x_train = (x_train - np.mean(x_train)) / np.std(x_train)  # Gaussian Normalization
x_test = (x_test - np.mean(x_test)) / np.std(x_test)



plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.show()


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation='elu', kernel_initializer=keras.initializers.GlorotUniform(seed=None)),
    keras.layers.Dropout(0.4),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(512, activation='relu', bias_initializer=initializers.Constant(0.1), kernel_initializer=keras.initializers.GlorotUniform(seed=None)),
    keras.layers.Dense(10)
])
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(x_train,  y_train, validation_split=0.1, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)
model.summary()



