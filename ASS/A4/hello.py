from data_utils import load_dataset
import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time
from scipy.linalg import cho_factor, cho_solve, pinv

Q1 = False
Q2 = True
random.seed(100)


def predict_y(weight, x):
    # print(x.shape)
    return np.dot(x, weight)


def get_loss(pred, actual):
    return np.sum((pred - actual) ** 2) / actual.shape[0]


def grad_descent(weight, x, y, lr):
    pred = predict_y(weight, x)
    # print(pred.shape)
    # print(x.shape)
    dw = np.dot(x.T, pred - y) / x.shape[0]

    weight = weight - dw * lr
    return weight


def grad_descent_moment(x, y, weight, lr, last, beta):
    pred = predict_y(weight, x)
    # print(pred.shape)
    # print(x.shape)
    if last is None:
        dw = np.dot(x.T, pred - y) / x.shape[0]
    else:
        dw = np.dot(x.T, pred - y) / x.shape[0] * (1 - beta) + last * beta

    weight = weight - dw * lr
    return weight, dw


def single_gradient(x, y, weight, lr):
    pred = predict_y(weight, x)
    dw = x * (pred - y)
    weight = weight - dw * lr
    return weight


def full_batch(x, y, epochs, lr, weight):
    losses = []
    losses.append(get_loss(predict_y(weight, x), y))
    # print(losses)
    timer = time.time()
    for i in range(epochs):
        weight = grad_descent(weight, x, y, lr)
        pred = predict_y(weight, x)
        loss = get_loss(pred, y)
        losses.append(loss)
        if loss < 0.7216 * 1.1:
            end_time = time.time() - timer
            epochy = i + 1

        # print(loss)
        print(weight)
    if loss > 0.7216 * 1.1:
        end_time = time.time() - timer
        epochy = i + 1
    return weight, losses, end_time, epochy


def mini_batch(x, y, epochs, lr, weight, batch_size):
    losses = []
    for i in range(epochs):
        x_batch = np.array([0])
        y_batch = np.array([0])
        timer = time.time()
        for j in range(batch_size):
            num = random.randint(0, 999)
            if x_batch.shape == (1,):
                x_batch = x[num, :]
                y_batch = y[num, :]
            else:
                # print(x_batch.shape)
                x_batch = np.vstack((x_batch, x[num, :]))
                y_batch = np.vstack((y_batch + y[num, :]))
        if batch_size == 1:
            x_batch, y_batch = x_batch.reshape(1, x_batch.shape[0]), y_batch.reshape(y_batch.shape[0], 1)
            weight = grad_descent(weight, x_batch, y_batch, lr)
        else:
            weight = grad_descent(weight, x_batch, y_batch, lr)

        pred = predict_y(weight, x)
        loss = get_loss(pred, y)
        losses.append(loss)
        if loss < 0.7216 * 1.1:
            end_time = time.time() - timer
            epochy = i + 1
    print(x_batch.shape)
    if loss > 0.7216 * 1.1:
        end_time = time.time() - timer
        epochy = i + 1
    return weight, losses, end_time, epochy


def mini_batch_moment(x, y, epochs, lr, weight, batch_size, beta):
    losses = []
    last = None
    for i in range(epochs):
        x_batch = np.array([0])
        y_batch = np.array([0])
        timer = time.time()

        for j in range(batch_size):
            num = random.randint(0, 999)
            if x_batch.shape == (1,):
                x_batch = x[num, :]
                y_batch = y[num, :]
        if batch_size == 1:
            x_batch, y_batch = x_batch.reshape(1, x_batch.shape[0]), y_batch.reshape(1, y_batch.shape[0])
            weight, last = grad_descent_moment(x_batch, y_batch, weight, lr, last, beta)

        pred = predict_y(weight, x)
        loss = get_loss(pred, y)
        losses.append(loss)
        if loss < 0.7216 * 1.1:
            end_time = time.time() - timer
            epochy = i + 1
    if loss > 0.7216 * 1.1:
        end_time = time.time() - timer
        epochy = i + 1
    print(x_batch.shape)
    return weight, losses, end_time, epochy


def testing_RMSE(x, y, weight):
    pred = predict_y(weight, x)
    error = np.sqrt(np.square(np.linalg.norm(pred - y, 2)) / y.shape[0])
    return np.sqrt(error)


def linear_regression(x_train, y_train, x_test):
    # add a column of ones to the data to account for the bias
    x_training = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    x_testing = np.hstack((np.ones((x_test.shape[0], 1)), x_test))

    # svd decomposition
    U, S, Vh = np.linalg.svd(x_training, full_matrices=False, compute_uv=True)
    weights = np.dot(np.dot(Vh.T, np.diag(1 / S)), np.dot(U.T, y_train))
    y_pred = np.dot(x_testing, weights)
    return y_pred


def plot_loss(losses, title):
    pred = linear_regression(x_train[:, 1:], y_train, x_test[:, 1:])
    real_losses = get_loss(pred, y_test)
    print(real_losses)
    for i in losses[0]:
        plt.plot(i)
    plt.plot([0.7216] * len(losses[0][0]), label='Exact')
    plt.legend(['0.3', '0.5', '0.7', '0.9', 'Exact'])
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    return None


if Q1:

    # initialize data
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
    x_train, y_train = x_train[:1000], y_train[:1000]
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
    bias = 0
    weights = np.zeros((x_train.shape[1], 1))
    beta = 0.9

    # full batch
    # weight, losses = full_batch(x_train, y_train, 100, 0.1, weights)
    # plot_loss(losses, 'Full Batch')
    # print('Full Batch RMSE: ', testing_RMSE(x_test, y_test, weight))

    # mini batch
    lr = [0.01, 0.001, 0.0001, 0.00001]
    beta = [0.3, 0.5, 0.7, 0.9]
    losseses = [[], []]
    for i in beta:
        bias = 0
        weights = np.zeros((x_train.shape[1], 1))
        weight, losses, timer, epochy = mini_batch_moment(x_train, y_train, 10000, 0.001, weights, 1, i)
        print('Beta: ', i)
        print('Time: ', timer)
        print('Epochs: ', epochy)
        print()
        losseses[0].append(losses)
        losseses[1].append(lr)
        # print('Mini Batch RMSE: ', testing_RMSE(x_test, y_test, weight))
    # weight, losses = mini_batch(x_train, y_train, 10000, 0.0006, weights, 10)
    plot_loss(losseses, 'SDG, Mini batch size = 1, lr = 0.001')
    print('Mini Batch RMSE: ', testing_RMSE(x_test, y_test, weight))

    # weight, losses = mini_batch(x_train, y_train, 20000, 0.0006, weights, 1)
    # plot_loss(losses, 'Mini Batch, batch size = 1')
    # print('Mini Batch RMSE: ', testing_RMSE(x_test, y_test, weight))

    # weight, losses = mini_batch(x_train, y_train, 20000, 0.004, weights, 1)
    # plot_loss(losses, 'Mini Batch')
    # print('Mini Batch RMSE: ', testing_RMSE(x_test, y_test, weight))

    # lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
    # losseses = [[],[]]
    # for i in lr:
    #     weight, losses = mini_batch(x_train, y_train, 1000, i, weights, 100)
    #     losseses[0].append(losses)
    #     losseses[1].append(lr)
    #     # print('Mini Batch RMSE: ', testing_RMSE(x_test, y_test, weight))
    # # weight, losses = mini_batch(x_train, y_train, 10000, 0.0006, weights, 10)
    # plot_loss(losseses, 'Mini Batch, Batch Size = 10')
if Q2:
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]
    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([y_train, y_valid])
    x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
    x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def log_likelyhood(x, y, weight):
        pred = sigmoid(np.dot(x, weight))
        return np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))


    def log_likelyhood_mini(x, y, weight):
        pred = sigmoid(x * weight)
        return np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))


    def gradient_log(x, y, weight):
        pred = sigmoid(np.dot(x, weight))
        return np.dot(x.T, pred - y)


    def gradient_log_mini(x, y, weight):
        pred = sigmoid(np.dot(x, weight))
        # print((x * (pred - y)).shape)
        return ((pred - y) * x).reshape(5, 1)


    def test_accuracy(pred, y):
        sum = 0
        for a, b in zip(pred, y):
            if a > 0.5:
                a = 1
            else:
                a = 0
            if a == b:
                sum += 1
        return sum / len(y)


    def full_batch_grad_descent(x, y, epoch, lr, weight):
        losses = []
        for i in range(epoch):
            loss = log_likelyhood(x, y, weight)
            losses.append(loss)
            weight = weight - lr * gradient_log(x, y, weight)
            # print(x.shape, y.shape, weight.shape)
        return weight, losses


    def mini_batch_gd(x, y, epoch, lr, weight, batch_size):
        losses = []
        for i in range(epoch):
            x_batch = np.array([0])
            y_batch = np.array([0])
            num = random.randint(0, x.shape[0] - 1)
            x_batch = x[num, :]
            y_batch = y[num]
            x_batch = x_batch.reshape(1, 5)
            y_batch = y_batch.reshape(1, 1)
            print(x_batch.shape, y_batch.shape, weight.shape)

            loss = log_likelyhood(x, y, weight)
            losses.append(loss)
            weight = weight - lr * gradient_log(x_batch, y_batch, weight)
        return weight, losses


    weights = np.zeros((x_train.shape[1], 1))
    weight, losses = mini_batch_gd(x_train, y_train, 3000, 0.01, weights, 1)
    actual_loss = [-x for x in losses]
    pred = sigmoid(np.dot(x_test, weight))
    print('Accuracy: ', test_accuracy(pred, y_test))
    print('Log likelyhood: ', log_likelyhood(x_test, y_test, weight))

    weights = np.zeros((x_train.shape[1], 1))
    weight, losses1 = full_batch_grad_descent(x_train, y_train, 3000, 0.01, weights)
    actual_loss1 = [-x for x in losses1]
    print(weight)
    plt.plot(actual_loss)
    plt.plot(actual_loss1)
    plt.title('Full Batch Gradient Descent')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()
    pred = sigmoid(np.dot(x_test, weight))
    print('Accuracy: ', test_accuracy(pred, y_test))
    print('Log likelyhood: ', log_likelyhood(x_test, y_test, weight))