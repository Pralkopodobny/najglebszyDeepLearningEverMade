import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization


def create_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer=tensorflow.keras.initializers.GlorotUniform(seed=None)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.20))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=tensorflow.keras.initializers.GlorotUniform(seed=None)))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax',
                    kernel_initializer=tensorflow.keras.initializers.GlorotUniform(seed=None)))

    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer=tensorflow.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model