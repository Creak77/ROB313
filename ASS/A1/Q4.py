import numpy as np
import time
from data_utils import load_dataset

def linear_regression(x_train, y_train, x_test):
    x_train = np.hstack([np.ones((x_train.shape[0], 1)), x_train])
    x_test = np.hstack([np.ones((x_test.shape[0], 1)), x_test])
    U, S, V = np.linalg.svd(x_train, full_matrices=False, compute_uv=True)
    w = V.T @ np.diag(1 / S) @ U.T @ y_train
    y_star = x_test @ w
    return y_star

if __name__ == '__main__':
    for dataset in ["mauna_loa", "rosenbrock", "pumadyn32nm", "iris", "mnist_small"]:
        print("_________________________________________________________________________________________________________")
        print(dataset)
        if dataset == 'rosenbrock':
            x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset, d=2, n_train=1000)
        else:
            x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset(dataset)
        x_train = np.vstack([x_valid, x_train])
        y_train = np.vstack([y_valid, y_train])
        t0 = time.time()
        y_star = linear_regression(x_train, y_train, x_test)
        if dataset in ["iris", "mnist_small"]:
            y_test_pred = np.argmax(y_star, axis=1)
            y_test = np.argmax(y_test, axis=1)
            accuracy = np.mean(y_test_pred == y_test)
            print("Test accuracy: {accuracy}".format(accuracy=round(accuracy, 6)))
        else:
            rmse = np.sqrt(np.mean(np.square(y_star - y_test)))
            print("Test RMSE: {rmse}".format(rmse=round(rmse, 6)))
        t1 = time.time()
        print("Time: {time}".format(time=round(t1 - t0, 6)))