import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils.mnist_reader as mnist_reader
import random


def translate(images, min_x=3, max_x=3, min_y=3, max_y=3,seed=None):
    if seed is not None:
        random.seed(seed)
    translated_images = np.empty(shape=images.shape)
    width = images.shape[1]
    height = images.shape[2]
    T = np.float32([[1, 0, random.randint(min_x, max_x)], [0, 1, random.randint(min_y, max_y)]])
    for i, image in enumerate(images):
        translated_images[i] = cv2.warpAffine(image, T, (width, height))
    return translated_images


def append(array1, array2):
    new_array = np.empty(shape=(array1.shape[0]+array2.shape[0], array1.shape[1], array1.shape[2]))
    for i, image in enumerate(array1):
        new_array[i] = image
    for i, image in enumerate(array2):
        new_array[i+array1.shape[0]] = image

    return new_array


def append_labels(labels1, labels2):
    new_array = np.empty(shape=(labels1.shape[0] + labels2.shape[0]))
    for i, label in enumerate(labels1):
        new_array[i] = label
    for i, label in enumerate(labels2):
        new_array[i+labels1.shape[0]] = label

    return new_array


def translate_test(images):
    images_translated = translate(images)
    plt.figure(figsize=(10, 10))
    for i in range(12):
        plt.subplot(6, 6, i * 2 + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.subplot(6, 6, i * 2 + 2)
        plt.imshow(images_translated[i])
    plt.show()


def append_test(images):
    images_translated = translate(images)
    meny_images = append(images, images_translated)
    for i in range(12):
        plt.subplot(6, 6, i * 2 + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(meny_images[i])
        plt.subplot(6, 6, i * 2 + 2)
        plt.imshow(meny_images[i+images.shape[0]])
    plt.show()


def append_labels_test(labels):
    meny_labels = append_labels(labels, labels)
    print(meny_labels.shape)
    for i in range(60000):
        if(meny_labels[i]!=meny_labels[60000+i]):
            print("error:", i)


if __name__ == "__main__":
    X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
    X_train = X_train.reshape((X_train.shape[0], 28, 28))
#    append_test(X_train)
    append_labels_test(y_train)
#    translate_test(X_train)