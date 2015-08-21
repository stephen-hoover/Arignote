"""
This module defines activation functions which provide nonlinearities for neural networks.
"""
from __future__ import division, print_function

import sys

from ..util import netlog

log = netlog.setup_logging("nnets_activations", level="INFO")

from theano import tensor as T


if sys.version_info.major == 2:
    string_types = (basestring,)
else:
    string_types = (str,)


def standardize_activation_name(activation):
    """ If activation functions have more than one name, this function standardizes them.
    """
    if activation in ["rec", "relu"]:
        activation = "relu"
    elif activation in ["rec_para", "prelu"]:
        activation = "prelu"

    return activation


def get_activation_func(activation):
    """Turns a string activation function name into a function.
    """
    if isinstance(activation, string_types):
        # Get the activation function.
        activation = activation.lower()
        if activation == "tanh":
            activation_func = tanh
        elif activation == "abstanh":
            activation_func = abs_tanh
        elif activation in ["sig", "sigmoid"]:
            activation_func = sigmoid
        elif activation in ["rec", "relu"]:
            activation_func = rectify
        elif activation in ["prelu_shelf"]:
            activation_func = parametric_flat_relu
        elif activation == "relu_max":
            activation_func = rectify_max  # For performance comparisons with abs version of rectify
        elif activation in ["rec_para", "prelu"]:
            activation_func = parametric_rectifier
        elif activation == "maxout":
            activation_func = maxout
        elif activation == "linear":
            activation_func = linear
        else:
            raise ValueError("Unrecognized activation: {}".format(activation))
    else:
        activation_func = activation

    return activation_func


# Alias tanh and sigmoid so that they're all defined in this module.
tanh = T.tanh
sigmoid = T.nnet.sigmoid


def linear(X):
    return X


def maxout(Xs):
    result = Xs[0]
    for X in Xs[1:]:
        result = T.maximum(result, X)
    return result


def parametric_rectifier(X, a):
    """
    Suggested in He et al. (2015): arXiv:1502.01852

    * `a` <Theano.shared>: A learnable parameter which controls
        the shape of this activation function.
    """
    return T.maximum(X, 0) - a * T.maximum(-X, 0)


def parametric_flat_relu(X, a, c):
    return T.maximum(X - c, 0) - a * T.maximum(-(X + c), 0)


def rectify(X):
    """Rectified linear activation function to provide non-linearity for NNs.
    Faster implementation using abs() suggested by Lasagne.
    """
    return (X + abs(X)) / 2


def rectify_max(X):
    """Rectified linear activation function to provide non-linearity for NNs. Function from
    https://github.com/Newmu/Theano-Tutorials/blob/master/5_convolutional_net.py
    """
    return T.maximum(X, 0.)


def abs_tanh(X):
    """Activation function. This is the absolute value of a hyperbolic tangent. It provides
    non-linearity which is constrained to always be positive.
    (I heard a claim that this works well for image recognition tasks.)
    """
    return abs(T.tanh(X))


def softmax(X):
    """Version of softmax claimed to be more numerically stable than Theano default. From
    https://github.com/Newmu/Theano-Tutorials/blob/master/5_convolutional_net.py
    This is also recommended in the Theano docs at
    http://deeplearning.net/software/theano/library/tensor/nnet/nnet.html#tensor.nnet.softmax
    """
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')
