import numpy as np
import matplotlib.pyplot as plt
from data_utils import load_dataset


def sigmoid(w, x):
    return 1./(np.ones((np.dot(x,w).shape[0],1)) + np.exp(np.dot(x,w)))

def f(x,y,w):
    first = np.vdot(y, (np.log(sigmoid(w, x))))
    temp1 = np.subtract(np.ones((y.shape[0], y.shape[1])), y)
    temp2 = np.log(np.ones((sigmoid(w, x).shape[0], sigmoid(w, x).shape[1]))-sigmoid(w, x))
    second = np.vdot(temp1, temp2)
    return -1. * (first + second)

def accuracy(y, y_pred):
    y_pred = np.rint(y_pred)
    result = 0
    for i in range(y_pred.shape[0]):
        if y[i] == y_pred[i]:
            result += 1
    return result * 1.0 / y_pred.shape[0]

def gd(x, y, f_pred):
    return (np.sum((y - f_pred)*x,axis=0)).reshape((5,1))

def GD(x_train, y_train, x_test, y_test, w, learning_rate, num_iter):
    k = 0
    losses, losses_test, accuracies, iters = np.array([]), np.array([]), np.array([]), np.array([])
    np.random.seed(100)
    while num_iter>0:
        f_pred = sigmoid(w, x_train)
        grad = gd(x_train, y_train, f_pred)
        w -= learning_rate*grad
        f_wp1 = f(x_train, y_train, w)
        losses = np.append(losses,f_wp1)
        losses_test = np.append(losses_test, f(x_test, y_test, w))
        y_pred_test = sigmoid(w, x_test)
        accuracies = np.append(accuracies, accuracy(y_test, y_pred_test))
        iters = np.append(iters, k)
        k += 1
        num_iter -= 1

    return w, losses, losses_test, accuracies, iters

def SGD(x_train, y_train, x_test, y_test, w, learning_rate, num_iter):
    k = 0
    losses, losses_test, accuracies, iters = np.array([]), np.array([]), np.array([]), np.array([])
    np.random.seed(100)
    while num_iter>0:
        t = np.random.randint(0, x_train.shape[0] - 1)
        x_t = x_train[t, :]
        f_pred = sigmoid(w, x_t)
        y_t = y_train[t]
        grad = gd(x_t, y_t, f_pred)
        w -= learning_rate*grad
        f_wp1 = f(x_train, y_train, w)
        losses = np.append(losses,f_wp1)
        losses_test = np.append(losses_test, f(x_test, y_test, w))
        y_pred_test = sigmoid(w, x_test)
        accuracies = np.append(accuracies, accuracy(y_test, y_pred_test))
        iters = np.append(iters, k)
        k += 1
        num_iter -= 1

    return w, losses, losses_test, accuracies, iters

def train(x_train, y_train, x_test, y_test, lrs, num_iter, select):
    for lr in lrs:
        w = np.zeros((x_train.shape[1],1))
        if select == 'GD':
            wk, losses, losses_test, accuracies, iters = SGD(x_train, y_train, x_test, y_test, w, lr, num_iter)
            plt.plot(losses, label="Learning rate {}".format(lr))
            print("Full Batch GD Training loss: {}".format(losses[losses.shape[0] - 1]))
            print("Full Batch GD Testing accuracy: {}, Loss: {}; learning rate = {}".format(
                accuracies[accuracies.shape[0] - 1], losses_test[losses_test.shape[0] - 1], lr))
        else:
            wk, losses, losses_test, accuracies, iters = GD(x_train, y_train, x_test, y_test, w, lr, num_iter)
            plt.plot(losses, label="Learning rate {}".format(lr))
            print("SGD Training loss: {}".format(losses[losses.shape[0] - 1]))
            print("SGD Testing accuracy: {}, Loss: {}; learning rate = {}".format(
                accuracies[accuracies.shape[0] - 1], losses_test[losses_test.shape[0] - 1], lr))
    if select == 'SGD':
        plt.xlabel("Iteration #")
        plt.ylabel("Negative Log Likelihood")
        plt.title("SGD Loss vs Iteration")
        plt.legend()
        plt.show()
    else:
        plt.xlabel("Iteration #")
        plt.ylabel("Negative Log Likelihood")
        plt.title("Full Batch GD Loss vs Iteration")
        plt.legend()
        plt.show()

x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
y_train = y_train[:,(1,)]
y_valid = y_valid[:,(1,)]
y_test = y_test[:,(1,)]

x_train = np.vstack((x_train,x_valid))
y_train = np.vstack((y_train,y_valid))

x_train = np.hstack((np.ones((x_train.shape[0],1)),x_train))
x_test = np.hstack((np.ones((x_test.shape[0],1)),x_test))

lr = [0.01,0.001,0.0001]
num_iter = 3000
w = np.zeros((x_train.shape[1],1))
print("Full Batch GD-------------")
train(x_train, y_train, x_test, y_test, lr, num_iter, 'GD')
print("Stochastic GD-------------")
train(x_train, y_train, x_test, y_test, lr, num_iter,'SGD')