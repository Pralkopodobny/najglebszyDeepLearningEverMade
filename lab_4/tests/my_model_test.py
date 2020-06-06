import tensorflow.keras
from models.myModel import create_model
from preprocessing.gaborFilter import gabor_filter
from preprocessing.binarizator import binarize
from preprocessing.translator import translate, append, append_labels


def raw_tests(batch_size=128, num_classes=10, epochs=30, img_rows=28, img_cols=28, filename="raw_model.h5"):
    #    1. step - load the data

    fashion_mnist = tensorflow.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    #    2. step - preprocess the data

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
    y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
    y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

    #   3,4 steps - define and compile the model

    model = create_model(input_shape, num_classes)

    #   5. step - fit the model

    model.fit(x_train, y_train,batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

    #   6. step - make the predictions

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #   7. step - save the model

    model.save(filename)


def gabor_tests(batch_size=128, num_classes=10, epochs=30, img_rows=28, img_cols=28, filename="gabor_model.h5"):
    #    1. step - load the data

    fashion_mnist = tensorflow.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    #    2. step - preprocess the data

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
    x_train = gabor_filter(x_train)
    x_test = gabor_filter(x_test)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
    y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
    y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

    #   3,4 steps - define and compile the model

    model = create_model(input_shape, num_classes)

    #   5. step - fit the model

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

    #   6. step - make the predictions

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #   7. step - save the model

    model.save(filename)


def binary_tests(batch_size=128, num_classes=10, epochs=30, img_rows=28, img_cols=28, filename="binary_model.h5"):
    #    1. step - load the data

    fashion_mnist = tensorflow.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    #    2. step - preprocess the data

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols)
    x_train = binarize(x_train, 0.1)
    x_test = binarize(x_test, 0.1)
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
    y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
    y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

    #   3,4 steps - define and compile the model

    model = create_model(input_shape, num_classes)

    #   5. step - fit the model

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

    #   6. step - make the predictions

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #   7. step - save the model

    model.save(filename)


def extension_tests(batch_size=128, num_classes=10, epochs=30, img_rows=28, img_cols=28, filename="extension_model.h5"):
    #    1. step - load the data

    fashion_mnist = tensorflow.keras.datasets.fashion_mnist
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    #    2. step - preprocess the data

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols)
    x_train = append(x_train, translate(x_train))
    y_train = append_labels(y_train, y_train)

    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
    x_train, x_test = x_train.astype('float32') / 255, x_test.astype('float32') / 255
    y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
    y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)

    #   3,4 steps - define and compile the model

    model = create_model(input_shape, num_classes)

    #   5. step - fit the model

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_split=0.1)

    #   6. step - make the predictions

    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #   7. step - save the model

    model.save(filename)

extension_tests()