
import numpy as np
from scipy.spatial import distance

def hamming_distance(X, X_train):
    """
    Zwróć odległość Hamminga dla obiektów ze zbioru *X* od obiektów z *X_train*.

    :param X: zbiór porównywanych obiektów N1xD
    :param X_train: zbiór obiektów do których porównujemy N2xD
    :return: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    """
    (N1, D) = X.shape
    (N2, D) = X_train.shape
    M = (np.ones(shape=(N1, D))-X)@X_train.transpose() + X@(np.ones(shape=(N2, D)) - X_train).transpose()
    return M
    pass

def euclidean_distance(X, X_train):
    return distance.cdist(X, X_train, metric='euclidean')

def sort_train_labels_knn(Dist, y):
    """
    Posortuj etykiety klas danych treningowych *y* względem prawdopodobieństw
    zawartych w macierzy *Dist*.

    :param Dist: macierz odległości pomiędzy obiektami z "X" i "X_train" N1xN2
    :param y: wektor etykiet o długości N2
    :return: macierz etykiet klas posortowana względem wartości podobieństw
        odpowiadającego wiersza macierzy Dist N1xN2

    Do sortowania użyj algorytmu mergesort.
    """
    """
    (N1, N2) = Dist.shape

    res = np.zeros(shape=Dist.shape)
    for x, r in enumerate(Dist):
        srt = np.argsort(r, kind="mergesort")
        res[x] = y[srt]
    return res
    """
    return y[np.argsort(Dist, kind="mergesort")]


def p_y_x_knn(y, k):
    """
    Wyznacz rozkład prawdopodobieństwa p(y|x) każdej z klas dla obiektów
    ze zbioru testowego wykorzystując klasyfikator KNN wyuczony na danych
    treningowych.

    :param y: macierz posortowanych etykiet dla danych treningowych N1xN2
    :param k: liczba najbliższych sasiadow dla KNN
    :return: macierz prawdopodobieństw p(y|x) dla obiektów z "X" N1xM
    """

    (N1, N2) = y.shape
    M = len(np.unique(y))
    y2 = y[:, :k]
    matrix = np.zeros(shape=(N1, M))
    for i, r in enumerate(y2):
        matrix[i] = np.bincount(r.astype(int), weights=None, minlength=M)
    matrix = matrix/k
    return matrix




def classification_error(p_y_x, y_true):
    """
    Wyznacz błąd klasyfikacji.

    :param p_y_x: macierz przewidywanych prawdopodobieństw - każdy wiersz
        macierzy reprezentuje rozkład p(y|x) NxM
    :param y_true: zbiór rzeczywistych etykiet klas 1xN
    :return: błąd klasyfikacji
    """

    (N, M) = p_y_x.shape

    args = M - np.argmax(p_y_x[::, ::-1], axis=1) - 1
    return np.count_nonzero(args - y_true) / N


def model_selection_knn_hamming(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param metric:
    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    best = [np.inf, 0, []]
    distance = hamming_distance(X_val, X_train)

    srt = sort_train_labels_knn(distance, y_train)
    for k in k_values:
        pyx = p_y_x_knn(srt, k)
        err = classification_error(pyx, y_val)
        if err < best[0]:
            best[0] = err
            best[1] = k
        best[2].append(err)
    return best


def model_selection_knn_euclidean(X_val, X_train, y_val, y_train, k_values):
    """
    Wylicz bład dla różnych wartości *k*. Dokonaj selekcji modelu KNN
    wyznaczając najlepszą wartość *k*, tj. taką, dla której wartość błędu jest
    najniższa.

    :param metric:
    :param X_val: zbiór danych walidacyjnych N1xD
    :param X_train: zbiór danych treningowych N2xD
    :param y_val: etykiety klas dla danych walidacyjnych 1xN1
    :param y_train: etykiety klas dla danych treningowych 1xN2
    :param k_values: wartości parametru k, które mają zostać sprawdzone
    :return: krotka (best_error, best_k, errors), gdzie "best_error" to
        najniższy osiągnięty błąd, "best_k" to "k" dla którego błąd był
        najniższy, a "errors" - lista wartości błędów dla kolejnych
        "k" z "k_values"
    """
    best = [np.inf, 0, []]
    distance = euclidean_distance(X_val, X_train)

    srt = sort_train_labels_knn(distance, y_train)
    for k in k_values:
        pyx = p_y_x_knn(srt, k)
        err = classification_error(pyx, y_val)
        if err < best[0]:
            best[0] = err
            best[1] = k
        best[2].append(err)
    return best
