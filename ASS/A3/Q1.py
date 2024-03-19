import numpy as np
import matplotlib.pyplot as plt
import random
from data_utils import load_dataset
import time


def rmse(y_1, y_2):
    return np.sqrt(np.average((y_1 - y_2) ** 2))

def mse_loss(y_pred, y_true):
    return np.mean(np.square(y_pred - y_true))

def linear_regression(x_train, x_valid, x_test, y_train, y_valid, y_test):
    x_total = np.vstack([x_train, x_valid])
    y_total = np.vstack([y_train, y_valid])
    x_total = x_total[:1000]
    y_total = y_total[:1000]
    X = np.ones((len(x_total), len(x_total[0]) + 1))
    X[:, 1:] = x_total
    U, S, Vh = np.linalg.svd(X)
    sig = np.diag(S)
    filler = np.zeros([len(x_total)-len(S), len(S)])
    sig_inv = np.linalg.pinv(np.vstack([sig, filler]))
    w = np.dot(Vh.T, np.dot(sig_inv, np.dot(U.T, y_total)))
    X_test = np.ones((len(x_test), len(x_test[0]) + 1))
    X_test[:, 1:] = x_test
    predictions = np.dot(X_test, w)
    result = rmse(y_test, predictions)
    return result, w

# Full-batch gradient descent
def full_batch_gradient_descent(x_train, y_train, x_test, y_test, learning_rates, epochs):
    loss_list = []
    for lr in learning_rates:
        #print(lr)
        w = np.zeros((x_train.shape[1], 1))
        loss_per_lr = []
        timer = time.time()
        k = 0
        for epoch in range(epochs):
            y_pred = np.dot(x_train, w)
            gradient = np.dot(x_train.T, y_pred - y_train) / x_train.shape[0]
            w -= lr * gradient
            y_pred_test = np.dot(x_test, w)
            loss = np.sum((y_pred - y_train) ** 2) / y_train.shape[0]
            loss_per_lr.append(loss)
            RMSE = rmse(y_test, y_pred_test)
        print(RMSE)
        loss_list.append(loss_per_lr)
    return loss_list

def single_gradient_momentum(x, y, w, lr, beta, last):
    pred = np.dot(x, w)
    if last is None:
        dw = np.dot(x.T, pred - y) / x.shape[0]
    else:
        dw = np.dot(x.T, pred - y) / x.shape[0] * (1 - beta) + last * beta
    w = w - dw * lr
    return w, dw

def mini_batch(x_train, y_train, x_test, y_test, learning_rates, epochs, batch_size):
    loss_list = []
    for lr in learning_rates:
        w = np.zeros((x_train.shape[1], 1))
        loss_per_lr = []
        for epoch in range(epochs):
            x_batch = np.array([0])
            y_batch = np.array([0])
            for j in range(batch_size):
                num = random.randint(0, 999)
                if x_batch.shape == (1,):
                    x_batch = x_train[num, :]
                    y_batch = y_train[num, :]
                else:
                    x_batch = np.vstack((x_batch, x_train[num, :]))
                    y_batch = np.vstack((y_batch, y_train[num, :]))
            y_pred = np.dot(x_batch, w)
            if batch_size == 1:
                x_batch, y_batch = x_batch.reshape(1, x_batch.shape[0]), y_batch.reshape(y_batch.shape[0], 1)
                gradient = np.dot(x_batch.T, y_pred - y_batch) / x_batch.shape[0]
                w -= lr * gradient
            else:
                gradient = np.dot(x_batch.T, y_pred - y_batch) / x_batch.shape[0]
                w -= lr * gradient
            y_pred_test = np.dot(x_test, w)
            y_pred_train = np.dot(x_train, w)
            loss = np.sum((y_pred_train - y_train) ** 2) / y_train.shape[0]
            loss_per_lr.append(loss)
            RMSE = rmse(y_test, y_pred_test)
        print(RMSE)
        loss_list.append(loss_per_lr)
    return loss_list

def stochastic_gradient_descent(x_train, y_train, x_test, y_test, learning_rates, epochs, batch_size, beta):
    loss_list = []
    for lr in learning_rates:
        w = np.zeros((x_train.shape[1], 1))
        loss_per_lr = []
        last = np.zeros((x_train.shape[1], 1))
        for epoch in range(epochs):
            x_batch = np.array([0])
            y_batch = np.array([0])
            for j in range(batch_size):
                num = random.randint(0, 999)
                if x_batch.shape == (1,):
                    x_batch = x_train[num, :]
                    y_batch = y_train[num, :]
                    x_batch, y_batch = x_batch.reshape(1, x_batch.shape[0]), y_batch.reshape(1, y_batch.shape[0])
                else:
                    x_batch = np.vstack((x_batch, x_train[num, :]))
                    y_batch = np.vstack((y_batch, y_train[num, :]))
            w, gradient = single_gradient_momentum(x_batch, y_batch, w, lr, beta, last)
            last = gradient
            y_pred_test = np.dot(x_test, w)
            y_pred_train = np.dot(x_train, w)
            loss = np.sum((y_pred_train - y_train) ** 2) / y_train.shape[0]
            loss_per_lr.append(loss)
            RMSE = rmse(y_test, y_pred_test)
        print(RMSE)
        loss_list.append(loss_per_lr)
    return loss_list

def generate_lossplots(x_train, y_train, opt_w):
    x_train = x_train[:1000]
    y_train = y_train[:1000]
    X = np.ones((len(x_train), len(x_train[0]) + 1))
    X[:, 1:] = x_train
    opt_pred = np.dot(X, opt_w)
    L_opt = 0
    for i in range(len(opt_pred)):
        L_opt += (opt_pred[i]-y_train[i])**2
    L_opt = L_opt / len(opt_pred)
    x_axis = list(range(1000))
    L = [L_opt] * 25000
    return L

np.random.seed(0)
x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('pumadyn32nm')
test_rmse, opt_w = linear_regression(x_train, x_valid, x_test, y_train, y_valid, y_test)
exact = generate_lossplots(x_train, y_train, opt_w)
x_train = x_train[:1000]
y_train = y_train[:1000]
x_train = np.hstack((np.ones((x_train.shape[0], 1)), x_train))
x_test = np.hstack((np.ones((x_test.shape[0], 1)), x_test))
learning_rates = [0.01, 0.001, 0.0001, 0.00001]

#beta = 0.9
beta = 0.7
#beta = 0.5
epochs = 25000
epoch_list = [i for i in range(epochs)]


# full batch
#loss_list = full_batch_gradient_descent(x_train, y_train, x_test, y_test, learning_rates, epochs)

# stochastic gradient descent with batch 1
loss_list = mini_batch(x_train, y_train, x_test, y_test, learning_rates, epochs, 1)
#loss_list1 = mini_batch(x_train, y_train, x_test, y_test, learning_rates, epochs, 1)

# stochastic gradient descent with batch 10
#loss_list = mini_batch(x_train, y_train, x_test, y_test, learning_rates, epochs, 10)
#loss_list2 = mini_batch(x_train, y_train, x_test, y_test, learning_rates, epochs, 10)

# stochastic gradient descent with batch 1 and momentum
#loss_list = stochastic_gradient_descent(x_train, y_train, x_test, y_test, learning_rates, epochs, 1, beta)
#loss_list3 = stochastic_gradient_descent(x_train, y_train, x_test, y_test, learning_rates, epochs, 1, beta)

#min_loss_epoch = np.argmin(loss_list[0])

# plot for first three
plt.plot(epoch_list, loss_list[0], label='learning rate = 0.01')
plt.plot(epoch_list, loss_list[1], label='learning rate = 0.001')
plt.plot(epoch_list, loss_list[2], label='learning rate = 0.0001')
plt.plot(epoch_list, loss_list[3], label='learning rate = 0.00001')

# plt.plot(epoch_list, loss_list1[1], label='learning rate = size 1')
# plt.plot(epoch_list, loss_list2[1], label='learning rate = size 10')
# plt.plot(epoch_list, loss_list3[1], label='learning rate = size 1 with momentum')

plt.plot(epoch_list, exact, label='exact')
plt.title("Epoch vs Loss at Various Learning Rates")
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(loc='best')
plt.show()

