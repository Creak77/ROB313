import autograd.numpy as np
from autograd.scipy.special import logsumexp
import autograd.numpy.random as npr
from autograd import grad
from autograd.misc.optimizers import adam
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# path to data
sys.path.append(os.path.join(os.getcwd(), "../../../data"))
from data_utils import load_dataset


def init_randn(m, n, rs=npr.RandomState(0)):
    """ init mxn matrix using small random normal
    """
    return 0.1 * rs.randn(m, n)


def init_xavier(m, n, rs=npr.RandomState(0)):
    """ TODO: init mxn matrix using Xavier intialization
    """
    return rs.randn(m, n)/np.sqrt(n)


def init_net_params(layer_sizes, init_fcn, rs=npr.RandomState(0)):
    """ inits a (weights, biases) tuples for all layers using the intialize function (init_fcn)"""
    return [
        (init_fcn(m, n), np.zeros(n))  # weight matrix  # bias vector
        for m, n in zip(layer_sizes[:-1], layer_sizes[1:])
    ]


def relu(x):
    return np.maximum(0, x)


def neural_net_predict(params, inputs):
    """Implements a deep neural network for classification.
       params is a list of (weights, bias) tuples.
       inputs is an (N x D) matrix.
       returns normalized class log-probabilities."""
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = relu(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)


def mean_log_like(params, inputs, targets):
    """ TODO: return the log-likelihood / the number of inputs 
    """
    return np.sum(neural_net_predict(params, inputs) * targets) / len(inputs)


def accuracy(params, inputs, targets, validation=False):
    """ return the accuracy of the neural network defined by params
    """
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    if validation:
        return np.mean(predicted_class == target_class), predicted_class
    return np.mean(predicted_class == target_class)

def get_matrix(predictions, target):
    cm = confusion_matrix(target, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()


if __name__ == "__main__":
    # ----------------------------------------------------------------------------------
    # EXAMPLE OF HOW TO USE STARTER CODE BELOW
    # ----------------------------------------------------------------------------------

    # loading data
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset("mnist_small")

    # initializing parameters
    layer_sizes = [784, 400, 10]
    params = init_net_params(layer_sizes, init_randn)

    # setting up training parameters
    num_epochs = 15
    learning_rate = 1e-2
    batch_size = 256

    # Constants for batching
    num_batches = int(np.ceil(len(x_train) / batch_size))
    rind = np.arange(len(x_train))
    npr.shuffle(rind)

    def batch_indices(iter):
        idx = iter % num_batches
        return rind[slice(idx * batch_size, (idx + 1) * batch_size)]

    # Define training objective
    def objective(params, iter):
        # get indices of data in batch
        idx = batch_indices(iter)
        return -mean_log_like(params, x_train[idx], y_train[idx])

    # Get gradient of objective using autograd.
    objective_grad = grad(objective)

    print("     Epoch     |    Train accuracy  |    Validation accuracy  |       Test accuracy  ")

    # Dictionary to store train/val history
    opt_history = {
        "train_nll": [],
        "val_nll": [],
        "train_loss": [],
        "val_loss": [],
    }

    def callback(params, iter, gradient):
        if iter % num_batches == 0:
            # record training & val. accuracy every epoch
            opt_history["train_nll"].append(-mean_log_like(params, x_train, y_train))
            opt_history["val_nll"].append(-mean_log_like(params, x_valid, y_valid))
            train_acc = accuracy(params, x_train, y_train, validation=False)
            val_acc, prediction = accuracy(params, x_valid, y_valid, validation=True)
            test_acc = accuracy(params, x_test, y_test, validation=False)
            print("{:15}|{:20}|{:20}|{:20}".format(iter // num_batches, train_acc, val_acc, test_acc))
            opt_history["train_loss"].append(1 - train_acc)
            opt_history["val_loss"].append(1 - val_acc)

    # We will optimize using Adam (a variant of SGD that makes better use of
    # gradient information).
    opt_params = adam(
        objective_grad,
        params,
        step_size=learning_rate,
        num_iters=num_epochs * num_batches,
        callback=callback,
    )

    val_acc, val_predictions = accuracy(opt_params, x_valid, y_valid, True)
    target = np.argmax(y_valid, axis=1)
    get_matrix(val_predictions, target)

    # Plotting the train and validation negative log-likelihood
    # plt.plot(opt_history["train_nll"])
    # plt.plot(opt_history["val_nll"])
    # plt.show()

    plt.plot(opt_history["train_loss"], label='Training loss')
    plt.plot(opt_history["val_loss"], label='Validation loss')
    plt.title('Training and Validation Accuracy Loss')
    plt.xlabel('Epoch number')
    plt.ylabel('Accuracy Loss')
    plt.legend(loc=0)
    plt.show()
