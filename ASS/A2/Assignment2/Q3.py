import numpy as np
from scipy.linalg import cho_factor, cho_solve
import time
from data_utils import load_dataset

def rbf_gaussian(x, z, theta):
    dist_sq = np.sum((x[:, np.newaxis, :] - z[np.newaxis, :, :])**2, axis=-1)
    K = np.exp(-dist_sq/theta)
    return K


if __name__ == '__main__':
    # # mauna_loa
    # x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    # thetas = np.array([0.05, 0.1, 0.5, 1, 2])
    # lambs = np.array([0.001, 0.01, 0.1, 1])
    #
    # loss = np.empty((len(thetas), len(lambs)))
    # J = np.identity(x_train.shape[0])
    # t0 = time.time()
    #
    # for i, theta in enumerate(thetas):
    #     for j, lamb in enumerate(lambs):
    #         K = rbf_gaussian(x_train, x_train, theta) + lamb * J
    #         L, lower = cho_factor(K)
    #         x = cho_solve((L, lower), y_train)
    #         y_star = rbf_gaussian(x_valid, x_train, theta).dot(x)
    #         loss[i, j] = np.linalg.norm(y_star - y_valid) / np.sqrt(y_valid.shape[0])
    #
    # best = np.unravel_index(loss.argmin(), loss.shape)
    # print("Best theta={}, lambda={}, loss={}".format(thetas[best[0]], lambs[best[1]], round(loss[best], 6)))
    # x_train = np.vstack([x_valid, x_train])
    # y_train = np.vstack([y_valid, y_train])
    # J = np.identity(x_train.shape[0])
    # K = rbf_gaussian(x_train, x_train, thetas[best[0]]) + lambs[best[1]] * J
    # L, lower = cho_factor(K)
    # x = cho_solve((L, lower), y_train)
    # y_star = rbf_gaussian(x_test, x_train, thetas[best[0]]).dot(x)
    # loss = np.linalg.norm(y_star - y_test) / np.sqrt(y_test.shape[0])
    # print("Test loss loss={}".format(round(loss, 6)))

    # # rosenbrock
    # x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock',d=2, n_train=1000)
    # thetas = np.array([0.05, 0.1, 0.5, 1, 2])
    # lambs = np.array([0.001, 0.01, 0.1, 1])
    #
    # loss = np.empty((len(thetas), len(lambs)))
    # J = np.identity(x_train.shape[0])
    # t0 = time.time()
    #
    # for i, theta in enumerate(thetas):
    #     for j, lamb in enumerate(lambs):
    #         K = rbf_gaussian(x_train, x_train, theta) + lamb * J
    #         L, lower = cho_factor(K)
    #         x = cho_solve((L, lower), y_train)
    #         y_star = rbf_gaussian(x_valid, x_train, theta).dot(x)
    #         loss[i, j] = np.linalg.norm(y_star - y_valid) / np.sqrt(y_valid.shape[0])
    #
    # best = np.unravel_index(loss.argmin(), loss.shape)
    # print("Best theta={}, lambda={}, loss={}".format(thetas[best[0]], lambs[best[1]], round(loss[best], 6)))
    # x_train = np.vstack([x_valid, x_train])
    # y_train = np.vstack([y_valid, y_train])
    # J = np.identity(x_train.shape[0])
    # K = rbf_gaussian(x_train, x_train, thetas[best[0]]) + lambs[best[1]] * J
    # L, lower = cho_factor(K)
    # x = cho_solve((L, lower), y_train)
    # y_star = rbf_gaussian(x_test, x_train, thetas[best[0]]).dot(x)
    # loss = np.linalg.norm(y_star - y_test) / np.sqrt(y_test.shape[0])
    # print("Test loss loss={}".format(round(loss, 6)))

    # iris
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    thetas = np.array([0.05, 0.1, 0.5, 1, 2])
    lambs = np.array([0.001, 0.01, 0.1, 1])
    acc = np.empty((thetas.shape[0], lambs.shape[0]))
    J = np.identity(x_train.shape[0])
    t0 = time.time()

    for i, theta in enumerate(thetas):
        for j, lamb in enumerate(lambs):
            K = rbf_gaussian(x_train, x_train, theta) + lamb * J
            L, lower = cho_factor(K)
            x = cho_solve((L, lower), y_train)
            y_star = rbf_gaussian(x_valid, x_train, theta).dot(x)
            acc[i, j] = np.sum(np.argmax(y_star, axis=1) == np.argmax(y_valid, axis=1)) / y_valid.shape[0]

    best = np.unravel_index(acc.argmax(), acc.shape)
    print("Best theta={}, lambda={}, acc={}".format(thetas[best[0]], lambs[best[1]], round(acc[best], 6)))
    x_train = np.vstack([x_valid, x_train])
    y_train = np.vstack([y_valid, y_train])
    J = np.identity(x_train.shape[0])
    K = rbf_gaussian(x_train, x_train, thetas[best[0]]) + lambs[best[1]] * J
    L, lower = cho_factor(K)
    x = cho_solve((L, lower), y_train)
    y_star = rbf_gaussian(x_test, x_train, thetas[best[0]]).dot(x)
    acc = np.sum(np.argmax(y_star, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Test acc acc={}".format(round(acc, 6)))
