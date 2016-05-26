'''
Module for multiclass classification via multiple outputs
Module Written by Anna Rafferty
'''
import numpy as np
import math

np.random.seed(123)


def logistic(x, T):
    """Logistic function"""
    return 1 /(1+np.exp(-T*x))
    
def dlogistic(x, T):
    """Derivative of the logistic function. Note that x has
    had the logistic function already applied to it."""
    return T * x*(1-x)

def propagate(x, w, b, f, T):
    """Propagate the net forward through one level, given the
        inputs (x), weights (w), bias term (b), and transfer function (f).
    """
    return f(np.dot(w, x) + b[:, None], T)

def error(t, y):
    """Computes the squared error between target values (t)
        and computed values (y).

        """
    return 0.5 * (t - y) ** 2
    
def derror(target, y):
    """Derivative of the error function."""
    return target-y

def our_error(weights, lambda1, lambda2):
    err = 0

    weight_sum = 0
    for layer in weights:
        for weight in layer[0]:
            i = weight ** 2
            j = (weight - 1) ** 2
            k = (weight + 1) ** 2
            weight_sum += lambda2 * i * j * k + lambda1 * weight**2
    err += weight_sum*0.5

    return err

def our_derror(target, computed, weights, lambda1, lambda2):
    d_error = derror(target, computed)
    weight_sum = 0
    for layer in weights:
        for weight in layer[0]:
            weight_sum += lambda1*weight + lambda2*(weight**2 - 1)*(3*weight**2 - 1)
    return np.add(d_error, weight_sum)

def update_weights(x, y, weights, bias, eta, T, lambda1, lambda2):
    """Update the weights of a 2-layer neural network, given the
        inputs (x), the expected outputs (y), a list of the weights at
        each layer (weights), a list of the biases at each layer (bias),
        a list of the transfer functions for each layer (funcs), and the
        learning rate (eta).

        This updates the weights in `weights` in place.

        """
    (w0, w1) = weights
    (b0, b1) = bias

    w0 += lambda1 * w0 + lambda2 * w0 * (w0**2 - 1) * (3 * w0**2 - 1)
    w1 += lambda1 * w1 + lambda2 * w1 * (w1**2 - 1) * (3 * w1**2 - 1)

    err = 0

    for i in range(x.shape[1]):
        # Propagate through the network
        x0 = x[:, [i]]
        x1 = propagate(x0, w0, b0, logistic, T)
        x2 = propagate(x1, w1, b1, logistic, T)

        err += np.sum(error(y[:, [i]], x2))

        # compute weight delta for the second layer
        d1 = derror(y[:, [i]], x2) * dlogistic(x2, T)

        a = np.dot(d1, x1.T)
        b = a + w1*lambda1

        dw1 = eta * b
        db1 = eta * d1.sum(axis=1)

        # compute weight delta for first layer
        d0 = np.dot(d1.T, w1).T * dlogistic(x1, T)
        a = np.dot(d0, x0.T)
        b = a + w0*lambda1
        dw0 = eta * b
        db0 = eta * d0.sum(axis=1)

        # update weights
        w0 += dw0
        w1 += dw1

        # update bias
        b0 += db0
        b1 += db1
#look here
        err += our_error(weights, lambda1, lambda2)


    return err

        

def train_multilayer_network(X, Y, weight_updating_function=update_weights, num_iters=50, num_hidden=12):
    """Function for training a network on input X to produce output
    Y. """

    # learning rate
    eta = 0.01

    T = 1     
    lambda1 = 0.00001
    lambda2 = 0

    # initialize the weights
    w0 = np.random.randn(num_hidden, X.shape[0])
    w1 = np.random.randn(Y.shape[0], num_hidden)
    weights = (w0, w1)

    # initialize the bias terms
    b0 = np.random.randn(num_hidden)
    b1 = np.random.randn(Y.shape[0])
    bias = (b0, b1)

    prev_error = float('inf')
    cur_error = float('inf')
    error_diff = 1
    reasonable_lambda = True
    
    #First stage of training
    while reasonable_lambda:
        while error_diff > 0.001:
            prev_error = cur_error
            cur_error = weight_updating_function(X, Y, weights, bias, eta, T, lambda1, lambda2)
            error_diff = prev_error - cur_error
        lambda1 *= 10
        T += 1
        error_diff = 1
        # Do one check outside loop to make sure jump is not too high
        cur_error = weight_updating_function(X, Y, weights, bias, eta, T, lambda1, lambda2)
        if prev_error*5 <= cur_error:
            reasonable_lambda = False

    print("done with first stage")
    #Second stage of training
    #get error back to previous ammount
    min_error = prev_error
    while cur_error > min_error:
        while error_diff > 0.001:
            prev_error = cur_error
            cur_error = weight_updating_function(X, Y, weights, bias, eta, T, lambda1, lambda2)
            error_diff = prev_error - cur_error
        lambda1 /= 2.0
        error_diff = 1

    return weights, bias


def predict_multilayer_network(X, weights, bias, hidden_layer_fn, output_layer_fn, T):
    """Fully propagate inputs through the entire neural network,
    given the inputs (X), a list of the weights for each level
    (weights), and a list of the bias terms for each level (bias).
    
    """
    Z = hidden_layer_fn(np.dot(weights[0], X) + bias[0][:, None], T)
    Y = output_layer_fn(np.dot(weights[1], Z) + bias[1][:, None], T)
    return Y

def get_confusion_matrix(network_output, desired_output):
    confusion_matrix = np.zeros([3,3])
    for i in range(len(network_output)):
        classification = np.argmax(network_output[i])
        real_output = np.argmax(desired_output[i])
        confusion_matrix[real_output][classification] += 1
    return confusion_matrix
