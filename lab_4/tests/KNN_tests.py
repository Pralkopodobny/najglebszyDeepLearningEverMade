import utils.mnist_reader as mnist_reader
from preprocessing.binarizator import binarize
from models.K_NN import model_selection_knn_euclidean, model_selection_knn_hamming
from datetime import datetime


def hamming_tests():

    X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
    X_train, X_test = X_train.astype('float32') / 255, X_test.astype('float32') / 255

    best_threshold = 0.1
    the_best_error = 100
    for i in range(1, 10):
        threshold = i / 10
        X_train_bin, X_test_bin = binarize(X_train, threshold), binarize(X_test, threshold)
        k_values = [3, 4, 5, 6, 7, 8, 9]
        error_best, best_k, errors = model_selection_knn_hamming(X_test_bin, X_train_bin, y_test, y_train, k_values)
        print("threshold", threshold, " acc", 1 - error_best, " best_k: ", best_k)
        if error_best < the_best_error:
            the_best_error = error_best
            best_threshold = threshold
    print("best acc: ", 1 - the_best_error)
    print("best threshold: ", best_threshold)


def euclidean_tests():
    X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
    X_train, X_test = X_train.astype('float32') / 255, X_test.astype('float32') / 255
    k_values = [3, 4, 5, 6, 7, 8, 9]
    print("StartTime =", datetime.now().strftime("%H:%M:%S"))
    error_best, best_k, errors = model_selection_knn_euclidean(X_test, X_train, y_test, y_train, k_values)
    print("EndTime =", datetime.now().strftime("%H:%M:%S"))
    print("error: ", error_best, "best_k: ", best_k)


if __name__ == "__main__":
    hamming_tests()
    euclidean_tests()
