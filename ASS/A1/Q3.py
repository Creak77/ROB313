from sklearn.neighbors import KDTree
from data_utils import load_dataset
import numpy as np
import time


def knn_classification_kdtree(x_train,y_train, x_test, k, l):
    y = np.empty((x_test.shape[0], 1))
    if (l == 1):
        tree = KDTree(x_train, metric='cityblock')
    elif (l == 2):
        tree = KDTree(x_train, metric='euclidean')
    else:
        pass
    ind = tree.query(x_test, k=k, return_distance=False, sort_results=False)
    for i, x in enumerate(x_test):
        vote, counts = np.unique(y_train[ind[i]], return_counts=True)
        if np.sum(counts == np.max(counts)) != 1:
            y[i, 0] = y_train[ind[0, 0]]
        else:
            y[i, 0] = vote[np.argmax(counts)]
    return y



if __name__ == '__main__':
    print("_________________________________________________________________________________________________________")
    print("iris")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    np.random.seed(0)
    y_train = np.tile(np.arange(y_train.shape[1]).reshape((1, -1)), (y_train.shape[0], 1))[y_train].reshape((-1, 1))
    y_valid = np.tile(np.arange(y_valid.shape[1]).reshape((1, -1)), (y_valid.shape[0], 1))[y_valid].reshape((-1, 1))
    y_test = np.tile(np.arange(y_test.shape[1]).reshape((1, -1)), (y_test.shape[0], 1))[y_test].reshape((-1, 1))
    n = int(x_train.shape[0])
    t0 = time.time()
    accuracy = np.empty((int(np.sqrt(n)), 2))
    for i in range(1, int(np.sqrt(n)) + 1):
        for j in range(1, 3):
            y_star = knn_classification_kdtree(x_train, y_train, x_valid, i, j)
            accuracy[i - 1, j - 1] = np.mean(y_star == y_valid)
            print("k={i}, l={j}, accuracy={accuracy}".format(i=i, j=j, accuracy=round(accuracy[i - 1, j - 1], 6)))
    best = np.unravel_index(np.argmax(accuracy), accuracy.shape)
    print("best params at k={i}, l={j} with accuracy={accuracy}".format(i=best[0] + 1, j=best[1] + 1,
                                                                        accuracy=round(accuracy[best], 6)))
    print("took {t}s".format(t=round(time.time() - t0, 2)))
    y_star = knn_classification_kdtree(x_train, y_train, x_test, best[0] + 1, best[1] + 1)
    print('test a ccuracy with best model: {accuracy}'.format(accuracy=round(np.mean(y_star == y_test), 6)))

    print("_________________________________________________________________________________________________________")
    print("mnist")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
    np.random.seed(0)
    y_train = np.tile(np.arange(y_train.shape[1]).reshape((1, -1)), (y_train.shape[0], 1))[y_train].reshape((-1, 1))
    y_valid = np.tile(np.arange(y_valid.shape[1]).reshape((1, -1)), (y_valid.shape[0], 1))[y_valid].reshape((-1, 1))
    y_test = np.tile(np.arange(y_test.shape[1]).reshape((1, -1)), (y_test.shape[0], 1))[y_test].reshape((-1, 1))
    n = int(x_train.shape[0])
    t0 = time.time()
    accuracy = np.empty((int(np.sqrt(n)), 2))
    for i in range(1, int(np.sqrt(n)) + 1):
        for j in range(1, 3):
            y_star = knn_classification_kdtree(x_train, y_train, x_valid, i, j)
            accuracy[i - 1, j - 1] = np.mean(y_star == y_valid)
            print("k={i}, l={j}, accuracy={accuracy}".format(i=i, j=j, accuracy=round(accuracy[i - 1, j - 1], 6)))
    best = np.unravel_index(np.argmax(accuracy), accuracy.shape)
    print("best params at k={i}, l={j} with accuracy={accuracy}".format(i=best[0] + 1, j=best[1] + 1,
                                                                        accuracy=round(accuracy[best], 6)))
    print("took {t}s".format(t=round(time.time() - t0, 2)))
    y_star = knn_classification_kdtree(x_train, y_train, x_test, best[0] + 1, best[1] + 1)
    print('test a ccuracy with best model: {accuracy}'.format(accuracy=round(np.mean(y_star == y_test), 6)))




