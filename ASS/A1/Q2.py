import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from data_utils import load_dataset
import numpy as np
import time

def knn_regression(x_train, y_train, x_test, k, l):
    y = np.empty((x_test.shape[0], 1))
    for i, x in enumerate(x_test):
        if l==1:
            dist = np.sum(np.abs(x_train-x.reshape((1,-1))), axis=1)
        elif l==2:
            dist = np.sqrt(np.sum(np.square(x_train-x.reshape((1,-1))), axis=1))
        else:
            pass
        k_nearest_neighbours = np.argpartition(dist, kth=k, axis=0)[:k]
        y[i,0] = np.average(y_train[k_nearest_neighbours, 0])
    return y

def knn_regression_kdtree(x_train,y_train, x_test, k, l):
    if (l == 1):
        tree = KDTree(x_train, metric='cityblock')
    elif (l == 2):
        tree = KDTree(x_train, metric='euclidean')
    else:
        pass
    ind = tree.query(x_test, k=k, return_distance=False, sort_results=False)
    y = np.mean(y_train[ind, 0], axis=1, keepdims=True)
    return y

if __name__ == '__main__':
    ds = np.array([2, 5, 10, 20, 50, 100, 200], dtype=int)
    kd_time = np.zeros(ds.shape)
    bf_time = np.zeros(ds.shape)
    for i, d in enumerate(ds):
        x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', d=d, n_train=5000)
        t0 = time.time()
        knn_regression(x_train, y_train, x_test, 5, 2)
        t1 = time.time()
        bf_time[i] = t1 - t0

        t0 = time.time()
        knn_regression_kdtree(x_train, y_train, x_test, 5, 2)
        t1 = time.time()
        kd_time[i] = t1 - t0

    plt.figure()
    plt.xlabel('d')
    plt.ylabel('time')
    plt.semilogy(ds, bf_time, label='brute force')
    plt.semilogy(ds, kd_time, label='kdtree')
    plt.legend(loc=0)
    plt.show()



