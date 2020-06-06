import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils.mnist_reader as mnist_reader


def gabor_filter(images, ksize=3, sigma=15, theta=1*np.pi/2, lambd=1*np.pi/4, gamma=0.5, psi=0):

    kernel = cv2.getGaborKernel((ksize,ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    images_filtered = np.empty(shape=images.shape)
    for i, image in enumerate(images):
        images_filtered[i] = cv2.filter2D(image, cv2.CV_8UC3, kernel)

    return images_filtered


def test_gabor(x_train, show_cv=False):
    x_train_filtered = gabor_filter(x_train)
    if show_cv ==True:
        for i in range(x_train.shape[0]):
            cv2.imshow('Oryginal Image', cv2.resize(x_train[i], (112, 112)))
            cv2.imshow('Filtered Image', cv2.resize(x_train_filtered[i], (112, 112)))
            cv2.waitKey()
            cv2.destroyAllWindows()
    else:
        plt.figure(figsize=(10, 10))
        for i in range(12):
            plt.subplot(6, 6, i*2+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(x_train[i])
            plt.subplot(6, 6, i*2 + 2)
            plt.imshow(x_train_filtered[i])
        plt.show()


def visualize_gabor(ksize=8, sigma=4, theta=1*np.pi/1, lambd=1*np.pi/4, gamma=0.5, psi=0):
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    cv2.imshow('Kernel', cv2.resize(kernel, (400, 400)))
    im = cv2.imread('../data/images/gabor_test_image.png')
    cv2.imshow('Oryginal Image', im)
    fim = cv2.filter2D(im, cv2.CV_8UC3, kernel)
    cv2.imshow('Filtered', fim)
    cv2.waitKey()


if __name__ == "__main__":
    X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
    X_train = X_train.reshape((X_train.shape[0], 28, 28))
    test_gabor(X_train)
#   visualize_gabor()

