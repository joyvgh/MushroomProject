'''
Module for multiclass classification via multiple outputs
Module Written by Anna Rafferty
Additons Written by Joy Hill and Valerie Lambert
Most original methods are reflected in their name, 
additions specified in comments. 
'''

import numpy as np
import math

np.random.seed(123)

def logistic(x, T):
    """Logistic function (by Anna) with slope T (by JV)"""
    return 1 /(1+np.exp(-T*x))
    
def dlogistic(x, T):
    """Derivative of the logistic function. Note that x has
    had the logistic function already applied to it."""
    return T * x*(1-x)

def propagate(x, w, b, f, T):
    """Propagate the net forward through one level, given the
        inputs (x), weights (w), bias term (b), and transfer function (f),
        and one parameter to transfer function (T). - Anna
    """
    return f(np.dot(w, x) + b[:, None], T)

def error(t, y):
    """Computes the squared error between target values (t)
        and computed values (y). - Anna
        """
    return 0.5 * (t - y) ** 2
    
def derror(target, y):
    """Derivative of the error function. - Anna"""
    return target-y

def our_error(weights, lambda1, lambda2):
    """Additional error used by our training algorithm - JV"""
    weight_sum = 0
    for layer in weights:
        for weight in layer[0]:
            i = weight ** 2
            j = (weight - 1) ** 2
            k = (weight + 1) ** 2
            weight_sum += lambda2 * i * j * k + lambda1 * weight**2
    weight_sum *= 0.5

    return weight_sum

def our_derror(target, computed, weights, lambda1, lambda2):
    """The derivative of the additional error used by our training algorithm - JV"""
    d_error = derror(target, computed)
    weight_sum = 0
    w0 = weights[0]
    for i in range(len(w0)):
        for j in range(len(w0[0])):
            weight_sum += lambda1*w0[0] + lambda2*(w0[0]**2 - 1)*(3*w0[0]**2 - 1)
    return np.add(d_error, weight_sum)

def update_weights(x, y, weights, bias, eta, T, lambda1, lambda2):
    """Update the weights of a 2-layer neural network, given the
        inputs (x), the expected outputs (y), a list of the weights at
        each layer (weights), a list of the biases at each layer (bias),
        a list of the transfer functions for each layer (funcs), and the
        learning rate (eta).

        This updates the weights in `weights` in place, and returns the
        total error of the network

        Minor additions written by JV

        """
    (w0, w1) = weights
    (b0, b1) = bias

    # Total error of the network
    err = 0

    for i in range(x.shape[1]):
        # Propagate through the network
        x0 = x[:, [i]]
        x1 = propagate(x0, w0, b0, logistic, T)
        x2 = propagate(x1, w1, b1, logistic, T)

        # Accumulate error for this set of x2
        err += np.sum(error(y[:, [i]], x2))

        # compute weight delta for the second layer
        d1 = derror(y[:, [i]], x2) * dlogistic(x2, T)

        # compute additional delta that penalizes large weights when lambda1 > 0
        a = np.dot(d1, x1.T)
        b = a + -1*w1*lambda1

        dw1 = eta * b
        db1 = eta * d1.sum(axis=1)

        # compute weight delta for first layer
        d0 = np.dot(d1.T, w1).T * dlogistic(x1, T)
        a = np.dot(d0, x0.T)

        # Additional delta penalizes large weights when lambda1 > 0
        # and penalizes non 0/1/-1 values when lambda2 > 0
        b = a + -1*w0*lambda1 + -1 * (3 * w0**5 - 4 * w0**3 + w0) * lambda2

        dw0 = eta * b
        db0 = eta * d0.sum(axis=1)

        # update weights
        w0 += dw0
        w1 += dw1

        # update bias
        b0 += db0
        b1 += db1

    # Accumulate error for new weights
    err += our_error(weights, lambda1, lambda2)

    return err

        

def train_multilayer_network(X, Y, weight_updating_function=update_weights, num_iters=50,
        num_hidden=1):
    """Function for training a network on input X to produce output
        Y using the algorithm described in notebook. Should produce
        a network with weights of 0, -1, and 1.
        Initialization and simple steps written by Anna. Everything 
        else is for the rule extraction algorithm written by JV.
    """

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

    # Set parameters for contorlling training
    prev_error = float('inf')
    cur_error = float('inf')
    error_diff = 1
    reasonable_lambda = True
    min_error = float('inf')
    
    # First stage of training
    # Train network with penalty to large weights

    while reasonable_lambda:

        # Train with lambda1 until there is only a small decrease of error
        while error_diff > 0.0001:
            prev_error = cur_error
            cur_error = weight_updating_function(X, Y, weights, bias, eta, T, lambda1, lambda2)
            error_diff = prev_error - cur_error

        # Once training slows, we need to check if changing lambda and T
        # increases our error by a factor of 5. If so first stage ends.
        cur_error -= our_error(weights, lambda1, lambda2)
        lambda1 *= 10
        T += 1
        cur_error += our_error(weights, lambda1, lambda2)
        if prev_error*5 <= cur_error or lambda1 > 0.1:
            reasonable_lambda = False

        # Otherwise reset parameter to train with new lambda and T
        error_diff = 1


    # Second stage of training
    # Get error back to previous ammount

    min_error = prev_error
    while cur_error > min_error and lambda1 <= 0.1:
        # Train with given lambda until training slows
        while error_diff > 0.0001:
            prev_error = cur_error
            cur_error = weight_updating_function(X, Y, weights, bias, eta, T, lambda1, lambda2)
            error_diff = prev_error - cur_error
        # Cut lambda in half if training slows
        lambda1 /= 2.0
        error_diff = 1

    # Third stage of training
    # Remove weights |W| < 0.1

    for i in range(len(w0)):
        for j in range(len(w0[0])):
            if abs(w0[i][j]) < 0.1:
                w0[i][j] = 0
    for i in range(len(w1[0])):
        if abs(w1[0][i]) < 0.1:
            w1[0][i] = 0

    # Fourth stage of training
    # Train weights to be near 0, 1, and -1

    lambda2 = 0.0001
    lambda1 = 0

    # Training stops once weights are within 0.05 of 0, 1, and -1
    while not weights_trained(weights):
        while error_diff > 0.0001:
            prev_error = cur_error
            cur_error = weight_updating_function(X, Y, weights, bias, eta, T, lambda1, lambda2)
            error_diff = prev_error - cur_error
        lambda2 *= 10
        T += 1
        error_diff = 1

    # Set weights to be exactly 0, 1, and -1.
    set_weights(weights)

    return weights, bias

def set_weights(weights):
    """Updates weights such that any weight within 0.05 of 0, 1, and -1
        is set to be 0, 1, and -1 respectively
    """
    w0 = weights[0]
    for i in range(len(w0)):
        for j in range(len(w0[0])):
            if abs(1 - w0[i][j]) < 0.05:
                w0[i][j] = 1
            elif abs(-1 - w0[i][j]) < 0.05:
                w0[i][j] = -1
            elif abs(w0[i][j]) < 0.05:
                w0[i][j] = 0

def weights_trained(weights):
    """Returns True if the weights are within 0.05 of 0, 1, and -1."""
    w0 = weights[0]
    w1 = weights[1]
    for i in range(len(w0)):
        for j in range(len(w0[0])):
            if abs(1 - w0[i][j]) > 0.05 and abs(-1 - w0[i][j]) > 0.05 and abs(w0[i][j]) > 0.05:
                return False
    return True

def predict_multilayer_network(X, weights, bias, hidden_layer_fn, output_layer_fn, T):
    """Fully propagate inputs through the entire neural network,
    given the inputs (X), a list of the weights for each level
    (weights), and a list of the bias terms for each level (bias).
    Anna.
    """
    Z = hidden_layer_fn(np.dot(weights[0], X) + bias[0][:, None], T)
    Y = output_layer_fn(np.dot(weights[1], Z) + bias[1][:, None], T)

    return Y

def get_confusion_matrix(network_output, desired_output):
    """Returns the confusion matrix for 3 outputs given 3 caluculated outputs and 3
        expected outputs. Written by Valerie.
    """
    confusion_matrix = np.zeros([3,3])
    for i in range(len(network_output)):
        classification = np.argmax(network_output[i])
        real_output = np.argmax(desired_output[i])
        confusion_matrix[real_output][classification] += 1
    return confusion_matrix
