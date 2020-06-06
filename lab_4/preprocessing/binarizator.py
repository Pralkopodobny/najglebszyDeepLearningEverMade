import numpy as np
import matplotlib.pyplot as plt
import utils.mnist_reader as mnist_reader


def binarize(images, threshold):
    binarized_images = np.empty(images.shape)

    for i, image in enumerate(images):
        binarized_images[i] = image > threshold

    return binarized_images


def binarize_test(images):
    binarized_images = binarize(images, 0.1)

    plt.figure(figsize=(10, 10))
    for i in range(12):
        plt.subplot(6, 6, i * 2 + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i])
        plt.subplot(6, 6, i * 2 + 2)
        plt.imshow(binarized_images[i])
    plt.show()


if __name__ == "__main__":
    X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
    X_train = X_train.reshape((X_train.shape[0], 28, 28))
    X_train = X_train.astype('float32') / 255
    binarize_test(X_train)