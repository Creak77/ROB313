from data_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt
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


def k_fold_cross_validation(x_train, y_train, folds):
    start_time = time.time()
    num_samples = x_train.shape[0]
    num_k = int(np.sqrt(num_samples))
    error_avg = np.empty((num_k, 2))
    for k in range(1, num_k + 1):
        rmse = np.empty((folds, 2))
        indices = -1 * np.ones(num_samples + folds - num_samples % folds, dtype=int)
        indices[:num_samples] = np.arange(num_samples, dtype=int)
        indices = np.random.permutation(indices).reshape((folds, -1))
        for fold in range(folds):
            valid_indices = indices[fold, indices[fold] >= 0]
            train_indices = np.delete(indices, fold, axis=0).reshape(-1)
            train_indices = train_indices[train_indices >= 0]
            x_train_cv = x_train[train_indices]
            y_train_cv = y_train[train_indices]
            x_valid_cv = x_train[valid_indices]
            y_valid_cv = y_train[valid_indices]
            for l in [1, 2]:
                y_star = knn_regression(x_train_cv, y_train_cv, x_valid_cv, k, l)
                rmse[fold, l - 1] = np.sqrt(np.mean(np.square(y_valid_cv - y_star)))
        error_avg[k - 1, 0] = np.average(rmse[:, 0])
        error_avg[k - 1, 1] = np.average(rmse[:, 1])
        print(f"k={k}, l=1, RMSE={round(error_avg[k - 1, 0], 6)}")
        print(f"k={k}, l=2, RMSE={round(error_avg[k - 1, 1], 6)}")
    best_params = np.unravel_index(np.argmin(error_avg), error_avg.shape)
    best_k, best_l = best_params[0] + 1, best_params[1] + 1
    best_error = round(error_avg[best_params[0], best_params[1]], 6)
    print(f"best params at k={best_k}, l={best_l} with RMSE={best_error}")
    print(f"took {round(time.time() - start_time, 2)}s")
    return error_avg, best_params

if __name__ == '__main__':
    print("_________________________________________________________________________________________________________")
    print("Mauna Loa Dataset")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mauna_loa')
    np.random.seed(0)
    x_train = np.vstack([x_valid, x_train])
    y_train = np.vstack([y_valid, y_train])
    e_avg, best = k_fold_cross_validation(x_train, y_train, 5)
    plt.figure()
    plt.plot(np.arange(1, int(np.sqrt(x_train.shape[0]))+1, 1), e_avg[:,1], label='l=2')
    plt.legend()
    plt.xlabel('k')
    plt.ylabel('RMSE')
    plt.title('validation error')
    plt.show()
    plt.figure()
    plt.title('test prediction with best model')
    plt.plot(np.concatenate((x_train, x_test)), np.concatenate((y_train, y_test)), label='actual')
    y_star = knn_regression(x_train, y_train, x_test, 2, 2)
    plt.plot(x_test, y_star, label='prediction k=2')
    print('test RMSE with k={k}: {RMSE}'.format(k=2, RMSE=round(np.sqrt(np.mean(np.square(y_test - y_star))), 6)))
    plt.legend(loc='best')
    plt.figure(figsize=(10, 6))
    plt.title('cross validation prediction')
    folds = 5
    n = int(x_train.shape[0])
    for k in ([2, 10]):
        y_plot = np.empty((0, 1))
        x_plot = np.empty((0, 1))
        indices = -1 * np.ones(n + folds - n % folds, dtype=int)
        indices[:n] = np.arange(n, dtype=int)
        indices = np.random.permutation(indices).reshape((folds, -1))
        for i in range(0, folds):
            valid_idx = indices[i, indices[i] >= 0]
            train_idx = np.delete(indices, i, axis=0).reshape(-1)
            train_idx = train_idx[train_idx >= 0]
            x_train_cv, x_valid_cv = x_train[train_idx], x_train[valid_idx]
            y_train_cv, y_valid_cv = y_train[train_idx], y_train[valid_idx]
            x_plot = np.concatenate((x_plot, x_valid_cv))
            y_star = knn_regression(x_train_cv, y_train_cv, x_valid_cv, k, 2)
            y_plot = np.concatenate((y_plot, y_star))
        plotset = np.array([x_plot, y_plot]).squeeze().T
        plotset = plotset[plotset[:, 0].argsort()]
        plt.plot(plotset[:, 0], plotset[:, 1], label="k={k}".format(k=k))
    plt.plot(x_train, y_train, label='actual')
    plt.legend(loc='best')
    plt.show()
    print("_________________________________________________________________________________________________________")
    print("pumadyn32nm")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
    np.random.seed(0)
    x_train = np.vstack([x_valid, x_train])
    y_train = np.vstack([y_valid, y_train])
    e_avg, best = k_fold_cross_validation(x_train, y_train, 5)
    y_star = knn_regression(x_train, y_train, x_test, best[0] + 1, best[1] + 1)
    print('test RMSE with best model: {RMSE}'.format(RMSE=round(np.sqrt(np.mean(np.square(y_test - y_star))), 6)))
    print("_________________________________________________________________________________________________________")
    print("rosenbrock")
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('rosenbrock', d=2, n_train=1000)
    np.random.seed(0)
    x_train = np.vstack([x_valid, x_train])
    y_train = np.vstack([y_valid, y_train])
    e_avg, best = k_fold_cross_validation(x_train, y_train, 5)
    y_star = knn_regression(x_train, y_train, x_test, best[0] + 1, best[1] + 1)
    print('test RMSE with best model: {RMSE}'.format(RMSE=round(np.sqrt(np.mean(np.square(y_test - y_star))), 6)))
    print("_________________________________________________________________________________________________________")


