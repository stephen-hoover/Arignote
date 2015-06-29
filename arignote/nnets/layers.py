"""
This module contains pieces of neural networks.
This code all started as copy-pastes from the University of Montreal's Deep Learning Tutorial:

http://deeplearning.net/tutorial/deeplearning.pdf

Each Layer class must define the following interface:

An instance must have the attributes
    * `input` : A Theano symbolic variable representing the layer's input
    * `output` : A Theano symbolic variable representing the layer's output
    * `n_in` : A tuple with the expected shape of an input array
    * `n_out` : A tuple with the expected shape of the output array
    * `l1` : L1 weight norm for use in regularization
    * `l2_sqr` : Square of the L2 weight norm for use in regularization
    * `params` : A list of all trainable symbolic parameters
    * `param_update_rules` : A list of dictionaries with e.g. max column norm rules.

The constructor must take the following inputs:
    * Positional arguments:
        `n_in` : Tuple, shape of the input
    * Optional arguments:
        `rng` <np.random.RandomState|None>
        `theano_rng` <theano.tensor.shared_randomstreams.RandomStreams|None>

The object must define the following functions:
    * `compile`, which takes a symbolic variable `input` and sets the layer outputs
    * `get_params`
    * `get_trainable_params`

Cuda-convnet optimizations:
http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
If the CUDA_CONVNET flag is set to True, we'll use Alex Krizhevsky's optimized GPU code
from the wrappers by pylearn2. The FilterActs from cuda-convnet has several limitations
compared to conv2d from Theano:

-- The number of channels must be even, or less than or equal to 3.
    If you want to compute the gradient, it should be divisible by 4.
    If you're training a convnet, that means valid numbers of
    input channels are 1, 2, 3, 4, 8, 12, 16, ...
-- Filters must be square, the number of rows and columns should be equal.
    For images, square filters are usually what you want anyway, but this
    can be a serious limitation when working with non-image data.
-- The number of filters must be a multiple of 16.
-- All minibatch sizes are supported, but the best performance is achieved
    when the minibatch size is a multiple of 128.
-- Only "valid" convolutions are supported. If you want to perform a "full" convolution,
    you will need to use zero-padding.
-- FilterActs only works on the GPU. We will fall back to conv2d when running on the CPU.
"""
from __future__ import division, print_function

from abc import ABCMeta

import numpy as np
import theano
from theano import tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams

from ..util import netlog
from ..nnets import activations as act
from ..util import misc


log = netlog.setup_logging("nnets_layers", level="INFO")


i_names = 0  # Counter to use for default layer names.


# For parametric rectified linear units, use this as the initial slope of the -x portion.
DEFAULT_PRELU_INIT = 0.25


# If we're running on a GPU, try to optimize convolutions with
# Alex Krizhevsky's cuda-convnet library as wrapped by pylearn2.
# Code snippets related to this taken from
# http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
CUDA_CONVNET = theano.config.device.startswith("gpu")
if CUDA_CONVNET:
    try:
        from pylearn2.sandbox.cuda_convnet import filter_acts
        from pylearn2.sandbox.cuda_convnet import pool
    except ImportError:
        log.error("Unable to import pylearn2; cuda-convnet optimization disabled.")
        CUDA_CONVNET = False
    from theano.sandbox.cuda.basic_ops import gpu_contiguous
if CUDA_CONVNET:
    log.info("Optimizing convolutions with code from cuda-convnet. Remember to convert "
             "inputs to c01b before convolutional layers and back to bc01 afterwards.")


def init_act_func_args(activation, n_units, layer_name):
    """ Get initialized parameters for an activation function.

    **Parameters**

    * `activation` <string>
        Standardized activation function name
    * `n_units` <int>
        Number of activation functions which need parameters
    * `layer_name` <string>
        Name of the layer which will use these parameters. Used to name the shared variables.

    **Returns**

    A tuple of Theano shared variables. If the activation function does not use
    parameters, the tuple will be empty.
    """
    args = ()
    if activation == "prelu":
        a = theano.shared(value=(DEFAULT_PRELU_INIT *
                          np.ones((n_units,), dtype=theano.config.floatX)),
                          name="a_{}".format(layer_name))
        args = (a,)
    elif activation == "prelu_shelf":
        a = theano.shared(value=(DEFAULT_PRELU_INIT *
                          np.ones((n_units,), dtype=theano.config.floatX)),
                          name="a_{}".format(layer_name))
        c = theano.shared(value=(0.05 *
                          np.ones((n_units,), dtype=theano.config.floatX)),
                          name="c_{}".format(layer_name))
        args = (a, c)

    return args


def get_weight_init(activation, shape, n_in, n_out, rng):
    """Initialize weights of a NN layer.

    Initial weight distributions should be chosen such that gradients neither blow up
    nor vanish as one progresses through the layers. For rectified linear and parametric
    rectified linear units, this derivation is in He et al. 2015 [1].

    Use the Xavier10 [2] suggested initialization for tanh and sigmoid activation functions.
    The Xavier paper suggests that initial weights be drawn from a uniform distribution between
    [sqrt(-6./(n_in+n_hidden)), sqrt(6./(n_in+n_hidden))] for a tanh activation function.
    For a logistic sigmoid, multiply this range by 4.

    The output of `uniform` is converted using `asarray` to dtype
    `theano.config.floatX` so that the code is runable on a GPU.

    [1] arXiv:1502.01852; http://arxiv.org/abs/1502.01852
    [2] http://machinelearning.wustl.edu/mlpapers/paper_files/AISTATS2010_GlorotB10.pdf
    """
    activation = act.get_activation_func(activation)  # Turn strings into functions.
    if activation in [act.rectify, act.parametric_rectifier, act.maxout, act.parametric_flat_relu]:
        # Use initialization recommended in He et al. 2015 for ReLU and PReLU activations.
        W_values = np.asarray(rng.normal(loc=0, scale=np.sqrt(2 / n_in),
                                         size=shape),
                              dtype=theano.config.floatX)
        if activation in [act.parametric_rectifier, act.parametric_flat_relu]:
            W_values /= np.sqrt(1 + DEFAULT_PRELU_INIT ** 2)
        elif activation == act.maxout:
            # Each portion of the maxout is linear, so initialize it as if it were a line.
            W_values /= np.sqrt(2)
    else:
        # Otherwise, use Xavier10 recomendations.
        if activation not in [act.tanh, act.sigmoid, act.abs_tanh]:
            log.warning("Using tanh - optimized weight initialization for this {} "
                            "activation function.".format(activation))
        W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)),
                                          high=np.sqrt(6. / (n_in + n_out)),
                                          size=shape),
                              dtype=theano.config.floatX)
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4

    return W_values


class Layer(object):
    """
    This is a generic layer of a neural network.

    **Optional Parameters**

    * `fix_params` <bool|False>
        If True, none of this layer's parameters will be updated during training.
        May also be a list of strings, in which case we'll only freeze parameters with
        names starting in one of the strings in the list.
    """
    __metaclass__ = ABCMeta

    def __init__(self, n_in, l1=0, l2=0, name=None, maxnorm=None,
                 fix_params=False, rng=None, theano_rng=None, may_train_unsupervised=True):
        self.input = None
        self.name = name
        self.n_in = n_in
        self.l1_lambda = l1
        self.l2_lambda = l2
        self.maxnorm = maxnorm
        self.fix_params = fix_params
        self.may_train_unsupervised = may_train_unsupervised

        if self.name is None:
            global i_names
            self.name = "lyr{}".format(i_names)
            i_names += 1

        # Initialize a random number generator, if not provided.
        if rng is None:
            log.debug("Making new Layer RNG in the {}".format(type(self)))
            rng = np.random.RandomState()
        # Create a Theano random generator that gives symbolic random values.
        if not theano_rng:
            theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.rng = rng
        self.theano_rng = theano_rng

        # Must also define the following in the subclass's __init__:
        self.n_out = None
        self.output = None
        self.output_inf_val = None

        self.set_l1_l2_norms(theano.shared(np.cast[theano.config.floatX](0), "zero"),
                             self.l1_lambda, self.l2_lambda)

        self.params = []
        self._param_update_rules = None  # Default to list of dicts specifying maxnorm
        self.activations = None  # Can create this after we run `compile`.

    def compile(self, input):
        """Use a symbolic input to set this layer's output.

        **Parameters**

        * `input` <theano.tensor.dmatrix>
            A symbolic tensor describing the Layer's input.

        **Modifies**

        `self.input`
        `self.output`
        `self.output_inf`
        """
        self.input = input

    def compile_activations(self, network_input):
        """Create a function which gives this Layer's activation, given inputs to its graph."""
        if self.output is None:
            raise AttributeError("The {} Layer {} hasn't been compiled "
                                 "yet.".format(type(self), self.name))
        self.activations = theano.function(inputs=[network_input], outputs=self.output)

    def set_l1_l2_norms(self, W, l1_lambda, l2_lambda):
        """
        Define the L1 and L2 penalties to be used in regularization.

        **Parameters**

        * `W` <theano.shared>
            Shared variable, assumed to represent weights.

        **Modifies**

        * `l1`
        * `l2_sqr`
        """
        if l1_lambda:
            self.l1 = l1_lambda * np.sum([np.abs(this_W).sum() for this_W in misc.flatten(W)])
        else:
            self.l1 = None
        if l2_lambda:
            self.l2_sqr = l2_lambda * np.sum([(this_W ** 2).sum() for this_W in misc.flatten(W)])
        else:
            self.l2_sqr = None

    # These definitions let us default to having the output for inference the
    # same as the regular output in our subclasses, but still write a new inference
    # output if we need to.
    def get_output_for_inference(self):
        return self.output if self.output_inf_val is None else self.output_inf_val
    def set_output_for_inference(self, new_output_inf):
        self.output_inf_val = new_output_inf
    output_inf = property(fget=get_output_for_inference, fset=set_output_for_inference)

    # These definitions let us default to having no special param_update_rules for this layer.
    # It must be the same length as `self.params`.
    def get_param_rules(self):
        if self._param_update_rules is None:
            rules = []
            for param in self.params:
                rules.append({"maxnorm": self.maxnorm})
                if self.fix_params is True or \
                      (self.fix_params and
                        any([param.name.startswith(fixname) for fixname in self.fix_params])):
                    rules[-1]["fixed"] = True
        else:
            rules = self._param_update_rules
        return rules

    def set_param_rules(self, new_param_update_rules):
        self._param_update_rules = new_param_update_rules
    param_update_rules = property(fget=get_param_rules, fset=set_param_rules)

    def get_params(self):
        """Returns a dictionary of parameters required by this class's __init__.
        """
        return dict(n_in=self.n_in, l1=self.l1_lambda, l2=self.l2_lambda,
                    fix_params=self.fix_params, name=self.name, maxnorm=self.maxnorm,
                    may_train_unsupervised=self.may_train_unsupervised)

    def _has_trainable_params(self):
        return len(self.get_trainable_params()) > 0
    has_trainable_params = property(_has_trainable_params)

    def get_trainable_params(self):
        """Returns a dictionary of parameters (the contents of this class's `params` list)
        which can be inserted into this class's __init__.
        """
        return dict()

    def set_trainable_params(self, params):
        """Given a dictionary of parameters, represented as numpy arrays or shared variables,
        replace the current values of those parameters in this instance by the input values.
        If the inputs are shared variables, set this Layer's parameters from the /values/ in
        those shared variables.
        """
        if isinstance(params, Layer):
            # Allow setting trainable parameters directly from another Layer.
            params = params.get_trainable_params()

        for name, val in params.items():
            this_param = getattr(self, name)
            if hasattr(val, "get_value"):
                # If `val` is a shared variable, grab its contents.
                val = val.get_value(borrow=True)

            if this_param is not None:
                if hasattr(this_param, "set_value"):
                    this_param.set_value(val, borrow=True)
                else:
                    for item, stored_item in zip(this_param, val):
                        if hasattr(stored_item, "get_value"):
                            stored_item = stored_item.get_value(borrow=True)
                        if hasattr(item, "set_value"):
                            item.set_value(stored_item, borrow=True)
                        else:
                            for sub_item, sub_stored in zip(item, stored_item):
                                if hasattr(sub_stored, "get_value"):
                                    sub_stored = sub_stored.get_value(borrow=True)
                                sub_item.set_value(sub_stored, borrow=True)

    def __getstate__(self):
        """Return a dictionary of this instance's contents suitable for pickling.
        This means primarily extracting the values stored in shared variables as numpy arrays.
        """
        state = self.get_params()
        state["trainable_params"] = {}
        for name, val in self.get_trainable_params().items():
            if val is None or (isinstance(val, (list, tuple)) and len(val) == 0):
                pkl_val = val
            elif hasattr(val, "get_value"):
                pkl_val = val.get_value(borrow=True)
            elif hasattr(val[0], "get_value"):
                pkl_val = [a.get_value(borrow=True) for a in val]
            elif hasattr(val[0][0], "get_value"):
                pkl_val = [[b.get_value(borrow=True) for b in a] for a in val]

            state["trainable_params"][name] = pkl_val

        return state

    def __setstate__(self, state):
        """Set the state of this Layer from a dictionary created by `__getstate__`.
        """
        trainable_params = state.pop("trainable_params")
        self.__init__(**state)

        self.set_trainable_params(trainable_params)  # Must be done after init.

    def __str__(self):
        return "{}: Neural network layer; input shape = {}".format(self.name, self.n_in)


class PassthroughLayer(Layer):
    def __init__(self, n_in, **kwargs):
        super(PassthroughLayer, self).__init__(n_in, **kwargs)
        self.n_out = n_in

    def compile(self, input):
        super(PassthroughLayer, self).compile(input)

        self.output = input


class InputLayer(PassthroughLayer):
    pass


class InputImageLayer(InputLayer):
    """This input layer reshapes a flat input to be a 3D array (multiple input
    channels of 2D images), suitable for passing to a convolutional layer.
    """
    def __init__(self, n_in, n_images, n_pixels, **kwargs):
        super(InputImageLayer, self).__init__(n_in, **kwargs)
        self.n_images = n_images
        self.n_pixels = n_pixels
        if not isinstance(n_images, int) or not getattr(n_pixels, "__len__", lambda: 0)() == 2:
            raise TypeError("The `n_images` input must be an integer, and the `n_pixels` "
                            "input must be a 2-tuple.")
        self.n_out = (n_images, n_pixels[0], n_pixels[1])

    def compile(self, input):
        super(InputImageLayer, self).compile(input)

        self.output = input.reshape((input.shape[0],) + self.n_out, ndim=4)

    def get_params(self):
        """Parameters required by __init__"""
        param_dict = super(InputImageLayer, self).get_params()
        param_dict.update(n_images=self.n_images, n_pixels=self.n_pixels)
        return param_dict

    def __str__(self):
        return "{}: Image-reshaping input layer; input shape = {}; output shape = {}".\
            format(self.name, self.n_in, self.n_out)


class DropoutLayer(PassthroughLayer):
    """Remove specified fraction of inputs. Adapted from
    https://github.com/Newmu/Theano-Tutorials/blob/master/5_convolutional_net.py

    Note that each example in a minibatch will have a different set of units dropped out;
    dropouts not only change between minibatches, they change between examples within a minibatch.
    """
    def __init__(self, n_in, dropout_p, theano_rng=None, **kwargs):
        """
        **Parameters**

        * `n_in` <int>
            Dimensionality of input
        * `dropout_p` <float>
            The fractional chance that a unit will be set to zero during each minibatch.

        **Optional Parameters**

        * `theano_rng` <theano.tensor.shared_randomstreams.RandomStreams|None>
            A symbolic random number generator
        * `name` <str|None>
            A designation for this layer. Can be used to retrieve it later.
        """
        super(DropoutLayer, self).__init__(n_in, theano_rng=theano_rng, **kwargs)
        self.dropout_p = dropout_p

    def compile(self, input):
        super(DropoutLayer, self).compile(input)

        if self.dropout_p > 0:
            retain_prob = 1 - self.dropout_p
            self.output = input * self.theano_rng.binomial(input.shape, p=retain_prob,
                                                           dtype=theano.config.floatX)
            self.output /= retain_prob

        self.output_inf = input  # Don't use any dropout during inference.

    def get_params(self):
        """Parameters required by __init__"""
        my_params = super(DropoutLayer, self).get_params()
        my_params["dropout_p"] = self.dropout_p
        return my_params

    def __str__(self):
        return "{}: Dropout layer with dropout probability {}".format(self.name, self.dropout_p)


class ClassificationOutputLayer(Layer):
    """Multi-class Logistic Regression Class
    The logistic regression is fully described by a weight matrix `W`
    and bias vector `b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """
    def __init__(self, n_in, n_classes, W=None, b=None, **kwargs):
        """ Initialize the parameters of the output layer.

        **Parameters**

        * `n_in` <int>
            Dimensionality of input
        * `n_classes` <int>
            Number of classes to predict

        **Optional Parameters**

        * `W` <theano.shared|None>
            Weights, shape (n_in, n_classes); will be initialized to zero if `None`.
        * `b` <theano.shared|None>
            Biases, shape (n_classes,); will be initialized to zero if `None`.
        * `name` <str|None>
            A designation for this layer. Can be used to retrieve it later.
        """
        if getattr(n_in, "__len__", None) is not None:
            if len(n_in) > 1:
                n_in = (np.prod(n_in),)  # Flatten a multi-D input
        else:
            raise TypeError("The input `n_in` must be a tuple.")

        super(ClassificationOutputLayer, self).__init__(n_in, **kwargs)
        self.n_classes = n_classes
        self.n_out = n_classes

        if W is None:
            # Initialize with 0 the weights W as a matrix of shape (n_in, n_out).
            # I find better results with 0 than with a random initialization. SH
            W = theano.shared(value=np.zeros((n_in[0], n_classes), dtype=theano.config.floatX),
                              name="W_{}".format(self.name), borrow=True)
        if b is None:
            # Initialize the biases b as a vector of n_out zeros.
            b = theano.shared(value=np.zeros((n_classes,), dtype=theano.config.floatX),
                              name="b_{}".format(self.name), borrow=True)
        self.W = W
        self.b = b

        self.set_l1_l2_norms(self.W, self.l1_lambda, self.l2_lambda)

        # Trainable parameters of the model:
        self.params = [self.W, self.b]

        self.p_y_given_x = None
        self.predict_proba = None
        self.y_pred = None

    def compile(self, input):
        input = input.flatten(2)  # Flatten a multi-D input
        super(ClassificationOutputLayer, self).compile(input)

        # Symbolic expression for computing the matrix of class-membership probabilities, where:
        # `W` is a matrix where column-k represent the separation hyper plain for class-k
        # `x` is a matrix where row-j represents input training sample-j
        # `b` is a vector where element-k represent the free parameter of hyperplane-k
        if self.n_classes == 1:
            self.p_y_given_x = act.sigmoid(T.dot(input, self.W) + self.b)
        else:
            self.p_y_given_x = act.softmax(T.dot(input, self.W) + self.b)
        self.output = self.p_y_given_x

        # Allow predicting on fresh features.
        self.predict_proba = theano.function(inputs=[self.input], outputs=self.p_y_given_x)

        # Symbolic description of how to compute prediction as class whose probability is maximal
        if self.n_classes == 1:
            self.y_pred = (self.p_y_given_x > 0.5)
        else:
            self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        **Math**

        \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
        \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
        \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
        \ell (\theta=\{W,b\}, \mathcal{D})

        **Parameters**

        * `y` <theano.tensor.TensorType>
            Corresponds to a vector that gives for each example the correct label.
            Note: we use the mean instead of the sum so that
            the learning rate is less dependent on the batch size.
        """
        #    y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch.
        #    T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1].
        #    T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP)
        # with one row per example and one column per class.
        #    LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0, y[0]], LP[1, y[1]], LP[2, y[2]], ..., LP[n-1, y[n-1]]], and
        #    T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch
        # examples) of the elements in v, i.e., the mean log-likelihood across the minibatch.
        if y.dtype.startswith("int"):
            if self.n_classes == 1:
                return -T.mean(y * T.log(self.p_y_given_x) + (1 - y) * T.log(1 - self.p_y_given_x), dtype=theano.config.floatX)
            else:
                return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y], dtype=theano.config.floatX)
        else:
            log.warning("Assuming that `y` is a 1D vector of ints designating class labels so I "
                        "can calculate NLL.")
            if self.n_classes == 1:
                return -T.mean(y * T.log(self.p_y_given_x) + (1 - y) * T.log(1 - self.p_y_given_x), dtype=theano.config.floatX)
            else:
                return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),
                                                       y.squeeze().astype("int32")],
                               dtype=theano.config.floatX)

    def error(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch: zero one
        loss over the size of the minibatch.

        **Parameters**

        * `y` <theano.tensor.TensorType>
            Corresponds to a vector that gives for each example the correct label.
        """
        y = T.flatten(y)
        ## Check if y has the same dimension as y_pred:
        #if y.ndim != self.y_pred.ndim:
        #    raise TypeError("y should have the same shape as self.y_pred",
        #                    ("y", y.type, "y_pred", self.y_pred.type))
        # Check if y is of the correct datatype:
        if y.dtype.startswith("int"):
            # The T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction.
            return T.mean(T.neq(self.y_pred, y), dtype=theano.config.floatX)
        else:
            log.warning("You've requested error with a float target. Rounding the targets.")
            return T.mean(T.neq(self.y_pred, T.round(y)), dtype=theano.config.floatX)

    def get_params(self):
        """Parameters required by __init__"""
        param_dict = super(ClassificationOutputLayer, self).get_params()
        param_dict.update(n_classes=self.n_out)
        return param_dict

    def get_trainable_params(self):
        """Returns a dictionary of parameters which can be inserted into this class's __init__.
        """
        return dict(W=self.W, b=self.b)

    def __str__(self):
        return "{}: Softmax output layer converting {} inputs to {} probabilities; " \
               "L1 penalty = {}, L2 penalty = {}, max column norm = {}".\
            format(self.name, self.n_in[0], self.n_classes,
                   self.l1_lambda, self.l2_lambda, self.maxnorm)


class OutputLayer(Layer):
    """Base class for output Layers.
    """
    def __init__(self, n_in, clip=None, **kwargs):
        """ Initialize the parameters of the output layer.

        **Parameters**

        * `n_in` <int>
            Dimensionality of input
        """
        if getattr(n_in, "__len__", None) is not None:
            if len(n_in) > 1:
                n_in = (np.prod(n_in),)  # Flatten a multi-D input
        else:
            raise TypeError("The input `n_in` must be a tuple.")

        super(OutputLayer, self).__init__(n_in, **kwargs)

        self.p_y_given_x = None
        self.predict_proba = None
        self.y_pred = None
        self.clip = clip
    #
    # def compile(self, input):
    #     input = input.flatten(2)  # Flatten a multi-D input
    #     super(OutputLayer, self).compile(input)
    #
    #     if self.clip is not None:
    #         input = T.clip(input, *self.clip)

    def error(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch: zero one
        loss over the size of the minibatch.

        **Parameters**

        * `y` <theano.tensor.TensorType>
            Corresponds to a vector that gives for each example the correct label.
        """
        # Check if y has the same dimension as y_pred:
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                            ("y", y.type, "y_pred", self.y_pred.type))
        # Check if y is of the correct datatype:
        if y.dtype.startswith("int"):
            # The T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction.
            return T.mean(T.neq(self.y_pred, y), dtype=theano.config.floatX)
        else:
            log.warning("You've requested error with a float target. Rounding the targets.")
            return T.mean(T.neq(self.y_pred, T.round(y)), dtype=theano.config.floatX)


class SoftmaxOutputLayer(OutputLayer):
    """Apply a softmax function to the inputs, and provide functions to calculate
    loss and classification accuracy.
    """
    def compile(self, input):
        input = input.flatten(2)  # Flatten a multi-D input
        super(SoftmaxOutputLayer, self).compile(input)

        if self.clip is not None:
            input = T.clip(input, *self.clip)
        self.p_y_given_x = act.softmax(input)

        # Allow predicting on fresh features.
        self.predict_proba = theano.function(inputs=[self.input], outputs=self.p_y_given_x)

        # Symbolic description of how to compute prediction as class whose probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        **Math**

        \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
        \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
        \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
        \ell (\theta=\{W,b\}, \mathcal{D})

        **Parameters**

        * `y` <theano.tensor.TensorType>
            Corresponds to a vector that gives for each example the correct label.
            Note: we use the mean instead of the sum so that
            the learning rate is less dependent on the batch size.
        """
        #    y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch.
        #    T.arange(y.shape[0]) is a symbolic vector which will contain [0,1,2,... n-1].
        #    T.log(self.p_y_given_x) is a matrix of Log-Probabilities (call it LP)
        # with one row per example and one column per class.
        #    LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0, y[0]], LP[1, y[1]], LP[2, y[2]], ..., LP[n-1, y[n-1]]], and
        #    T.mean(LP[T.arange(y.shape[0]),y]) is the mean (across minibatch
        # examples) of the elements in v, i.e., the mean log-likelihood across the minibatch.
        if y.dtype.startswith("int"):
            return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y], dtype=theano.config.floatX)
        else:
            log.warning("Assuming that `y` is a 1D vector of ints designating class labels so "
                        "I can calculate NLL.")
            return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]),
                                                   T.round(y).squeeze()],
                           dtype=theano.config.floatX)

    def __str__(self):
        return "{}: Softmax output layer for {} outputs".format(self.name, self.n_in[0])


class OrdinalOutputLayer(OutputLayer):
    """The number of the output class has meaning -- 2 is closer to 3 than to 0.
    """

    def compile(self, input):
        input = input.flatten(2)  # Flatten a multi-D input
        super(OrdinalOutputLayer, self).compile(input)

        if self.clip is not None:
            input = T.clip(input, *self.clip)
        self.p_y_given_x = act.softmax(input)

        # Allow predicting on fresh features.
        self.predict_proba = theano.function(inputs=[self.input], outputs=self.p_y_given_x)

        # Symbolic description of how to compute prediction as class whose probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)


class MultilabelOutputLayer(Layer):
    """Consider each label as an independent probabilitiy from zero to one, and
    provide functions to calculate loss and classification accuracy.
    """
    def __init__(self, n_in, **kwargs):
        """ Initialize the parameters of the output layer.

        **Parameters**

        * `input` <theano.tensor.dmatrix>
            A symbolic tensor of shape (n_examples, n_in)
        * `n_in` <int>
            Dimensionality of input
        """
        if getattr(n_in, "__len__", None) is not None:
            if len(n_in) > 1:
                n_in = (np.prod(n_in),)  # Flatten a multi-D input
        else:
            raise TypeError("The input `n_in` must be a tuple.")

        super(MultilabelOutputLayer, self).__init__(n_in, **kwargs)

        self.p_y_given_x = None
        self.predict_proba = None
        self.y_pred = None

    def compile(self, input):
        input = input.flatten(2)  # Flatten a multi-D input
        super(MultilabelOutputLayer, self).compile(input)

        self.p_y_given_x = act.sigmoid(self.input)

        # Allow predicting on fresh features.
        self.predict_proba = theano.function(inputs=[self.input], outputs=self.p_y_given_x)

        # Symbolic description of how to compute prediction as class whose probability is maximal
        self.y_pred = T.round(self.p_y_given_x) #(self.p_y_given_x > 0.5)

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        **Parameters**

        * `y` <theano.tensor.TensorType>
            Corresponds to an array that gives the correct label for each label in each example.
            Note: we use the mean instead of the sum so that
            the learning rate is less dependent on the batch size.
        """
        return -T.mean(y * T.log(self.p_y_given_x) +
                       (1 - y) * T.log(1 - self.p_y_given_x), dtype=theano.config.floatX)

    def error(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch: zero one
        loss over the size of the minibatch.

        **Parameters**

        * `y` <theano.tensor.TensorType>
            Corresponds to a vector that gives for each example the correct label.
        """
        # Check if y has the same dimension as y_pred:
        if y.ndim != self.y_pred.ndim:
            raise TypeError("y should have the same shape as self.y_pred",
                            ("y", y.type, "y_pred", self.y_pred.type))
        # Check if y is of the correct datatype:
        if y.dtype.startswith("int"):
            # The T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction.
            return T.mean(T.neq(self.y_pred, y), dtype=theano.config.floatX)
        else:
            log.warning("You've requested error with a float target. Rounding the targets.")
            return T.mean(T.neq(self.y_pred, T.round(y)), dtype=theano.config.floatX)

    def __str__(self):
        return "{}: Multilabel output layer for {} outputs".format(self.name, self.n_in[0])


class LCLayer(Layer):
    def __init__(self, n_in, n_units, n_segments, activation="prelu",
                 W=None, b=None, bvis=None, act_func_args=None, act_func_args_vis=None,
                 corruption_level=0.2, **kwargs):
        """A "locally connected", as opposed to "fully connected" layer.
        This layer connects subportions of the input layer to discrete sets of neurons.
        The input will be divided into equal portions, each given a different weight matrix.


        Hidden unit activation will be given by: activation(dot(input, W) + b).

        **Parameters**

        * `n_in` <int>
            Dimensionality of input
        * `n_units` <int>
            Number of hidden units
        * `n_segments` <int>
            Create this many equal segments, each with `n_units` neurons.

        **Optional Parameters**

        * `name` <str|None>
            A designation for this layer. Can be used to retrieve it later.
        * `activation` <str|"prelu">
            The activation function to use for this layer's outputs.
        * `W` <list of theano.shared|None>
            Weights, shape (n_in, n_units); will be initialized randomly if `None`.
        * `b` <list of theano.shared|None>
            Biases, shape (n_units,); will be initialized to zero if `None`.
        * `act_func_args` <tuple of theano.shared|None>
            Arguments for activation functions with trainable parameters, e.g. the slope
            of a PReLU activation function.
        * `act_func_args_vis` <tuple of theano.shared|None>
            As the `act_func_args`, but for reconstructed visible units during autoencoder training.
        * `rng` <numpy.random.RandomState|None>
            A random number generator; not used in this class
        * `theano_rng` <theano.tensor.shared_randomstreams.RandomStreams|None>
            A symbolic random number generator; not used in this class
        """
        if getattr(n_in, "__len__", None) is not None:
            if len(n_in) > 1:
                n_in = (np.prod(n_in),)  # Flatten a multi-D input
        else:
            raise TypeError("The input `n_in` must be a tuple.")

        super(LCLayer, self).__init__(n_in, **kwargs)
        self.n_units = n_units
        self.n_segments = n_segments
        self.n_out = (n_segments * n_units,)
        self.activation = act.standardize_activation_name(activation)
        self.activation_func = act.get_activation_func(activation)  # Turn strings into functions.
        self.corruption_level = corruption_level

        ######################################
        # Initialize the trainable parameters.

        # Divide the input up into even-ish segments.
        self.input_segments = []
        seg_len = self.n_in[0] // self.n_segments
        self.n_in_segments = []
        for i_seg in range(self.n_segments):
            seg_start = i_seg * seg_len
            seg_end = seg_start + seg_len
            if i_seg == self.n_segments - 1:
                seg_end = self.n_in[0]  # Go to the end of the input with the last segment.
            self.n_in_segments.append(seg_end-seg_start)

        if W is None:
            W = []
            for i_seg, n_seg in enumerate(self.n_in_segments):
                W.append(theano.shared(value=get_weight_init(self.activation, (n_seg, n_units),
                                                    n_seg, n_units, self.rng),
                                        name="W_{}_{}".format(self.name, i_seg), borrow=True))
        if b is None:
            b = []
            for i_seg, n_seg in enumerate(self.n_in_segments):
                b_values = np.zeros((n_units,), dtype=theano.config.floatX)
                b.append(theano.shared(value=b_values,
                                       name="b_{}_{}".format(self.name, i_seg), borrow=True))
        if act_func_args is None:
            act_func_args = []
            for i_seg, n_seg in enumerate(self.n_in_segments):
                act_func_args.append(init_act_func_args(self.activation, n_units,
                                                        "{}_{}".format(self.name, i_seg)))

        self.W = W
        self.b = b
        self.act_func_args = act_func_args

        # Use these parameters only if training as an autoencoder.
        # We'll initialize `bvis` to zeros only if we need it.
        self.bvis = bvis  # bvis corresponds to the bias of the visible units
        self.W_prime = [W.T for W in self.W]  # Use tied weights, therefore W_prime is W transpose.
        self.act_func_args_vis = act_func_args_vis
        self.ac_params = []

        # Define regularization parameters.
        self.set_l1_l2_norms(self.W, self.l1_lambda, self.l2_lambda)

        # Store the parameters of the model.
        self.params = self.W + self.b
        for args in self.act_func_args:
            self.params.extend(args)

    def compile(self, input):
        input = input.flatten(2)  # Flatten a multi-D input
        super(LCLayer, self).compile(input)

        output_segments = []
        seg_start = 0
        for i_seg, n_seg in enumerate(self.n_in_segments):
            self.input_segments.append(self.input[:, seg_start: seg_start + n_seg])
            seg_start += n_seg

            lin_output = T.dot(self.input_segments[i_seg], self.W[i_seg]) + self.b[i_seg]
            output_segments.append(self.activation_func(lin_output, *self.act_func_args[i_seg]))
        self.output = T.concatenate(output_segments, axis=1)

    def _get_corrupted_input(self, input_segments, corruption_level):
        """This function keeps `1-corruption_level` entries of the inputs the
        same and zero-out randomly selected subset of size `coruption_level`
        Note : The first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
            The second argument is the number of trials
            The third argument is the probability of success of any trial

            This will produce an array of 0s and 1s where 1 has a
                probability of 1 - `corruption_level` and 0 with
                `corruption_level`

                The binomial function returns the int64 data type by
                default.  int64 multiplied by the input
                type(floatX) always returns float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, so this doesn't change the
                result. This is needed to allow the GPU to work
                correctly as it only supports float32 for now.

        This function exists to support training this class as a denoising autoencoder.
        """
        return [self.theano_rng.binomial(size=inp.shape, n=1,
                                         p=1 - corruption_level,
                                         dtype=theano.config.floatX) * inp
                for inp in input_segments]

    def _get_hidden_values(self, input_segments):
        """ Computes the values of the hidden layer. """
        output_segments = []
        for i_seg in range(self.n_segments):
            lin_output = T.dot(input_segments[i_seg], self.W[i_seg]) + self.b[i_seg]
            output_segments.append(self.activation_func(lin_output, *self.act_func_args[i_seg]))

        return output_segments

    def _get_reconstructed_input(self, hidden_segments):
        """Computes the reconstructed visible values given the values of the hidden layer.
        """
        recon_segments = []
        for i_seg in range(self.n_segments):
            lin_output = T.dot(hidden_segments[i_seg], self.W_prime[i_seg]) + self.bvis[i_seg]
            recon_segments.append(self.activation_func(lin_output, *self.act_func_args_vis[i_seg]))

        return recon_segments

    def get_autoencoder_loss(self, corruption_level, regularized=False, loss="squared"):
        """ This function computes the cost and the updates for one training
        step of the dA. """
        if not self.bvis:
            # We'll only initialize this if we need it.
            self.bvis = []
            #seg_len = self.n_in[0] // self.n_segments
            for i_seg, n_seg in enumerate(self.n_in_segments):
                #if i_seg == len(self.n_in_segments) - 1:
                #    # The last segment might be short.
                #    last_seg_len = self.n_in[0] % seg_len
                #    seg_len = seg_len if last_seg_len == 0 else last_seg_len
                b_values = np.zeros((n_seg,), dtype=theano.config.floatX)
                self.bvis.append(theano.shared(value=b_values,
                                               name="bvis_{}_{}".format(self.name, i_seg),
                                               borrow=True))
        if not self.act_func_args_vis:
            self.act_func_args_vis = []
            #seg_len = self.n_in[0] // self.n_segments
            for i_seg, n_seg in enumerate(self.n_in_segments):
                #if i_seg == len(self.n_in_segments) - 1:
                #    # The last segment might be short.
                #    last_seg_len = self.n_in[0] % seg_len
                #    seg_len = seg_len if last_seg_len == 0 else last_seg_len
                self.act_func_args_vis.append(init_act_func_args(self.activation, n_seg,
                                                        "vis_{}_{}".format(self.name, i_seg)))
        if not self.ac_params:
            self.ac_params = self.params.copy() + self.bvis
            for args in self.act_func_args_vis:
                self.ac_params.extend(args)

        tilde_x = self._get_corrupted_input(self.input_segments, corruption_level)
        y = self._get_hidden_values(tilde_x)  # Returns a list of segments.
        z = T.concatenate(self._get_reconstructed_input(y), axis=1)

        # Note : We sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        if loss == "cross-entropy":
            minibatch_loss = - T.sum(self.input * T.log(z) + (1 - self.input) * T.log(1 - z), axis=1)
        elif loss == "squared":
            minibatch_loss = T.sum((self.input - z) ** 2, axis=1)
        else:
            raise ValueError("Unrecognized loss function: \"{}\".".format(loss))

        # Note : `minibatch_loss` is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch.
        loss = T.mean(minibatch_loss)
        if regularized:
            loss = loss + self.l1 + self.l2_sqr

        return loss

    def get_params(self):
        """Parameters required by __init__"""
        param_dict = super(LCLayer, self).get_params()
        param_dict.update(n_units=self.n_units, n_segments=self.n_segments,
                          activation=self.activation, corruption_level=self.corruption_level)
        return param_dict

    def get_trainable_params(self):
        """Returns a dictionary of parameters which can be inserted into this class's __init__.
        """
        return dict(W=self.W, b=self.b, bvis=self.bvis,
                    act_func_args=self.act_func_args, act_func_args_vis=self.act_func_args_vis)

    def __str__(self):
        return "{}: Locally-connected flat layer with {} segments of {} neurons using {} " \
               "activation;\n\tLayer input shape = {}; output shape = {}; L1 penalty = {}, " \
               "L2 penalty = {}, max column norm = {}".\
            format(self.name, self.n_segments, self.n_units, self.activation, self.n_in, self.n_out,
                   self.l1_lambda, self.l2_lambda, self.maxnorm)


SegmentedLayer = LCLayer  # I used to call the "LCLayer" a "SegmentedLayer".


class LCMaxoutLayer(Layer):
    def __init__(self, n_in, n_units, n_segments, maxout_k=5,
                 W=None, b=None, **kwargs):
        """A "locally connected", as opposed to "fully connected" layer.
        This version uses a maxout activation function.
        This layer connects subportions of the input layer to discrete sets of neurons.
        The input will be divided into equal portions, each given a different weight matrix.


        Hidden unit activation will be given by: activation(dot(input, W) + b).

        **Parameters**

        * `n_in` <int>
            Dimensionality of input
        * `n_units` <int>
            Number of hidden units
        * `n_segments` <int>
            Create this many equal segments, each with `n_units` neurons.

        **Optional Parameters**

        * `name` <str|None>
            A designation for this layer. Can be used to retrieve it later.
        * `W` <list of theano.shared|None>
            Weights, shape (n_in, n_units); will be initialized randomly if `None`.
        * `b` <list of theano.shared|None>
            Biases, shape (n_units,); will be initialized to zero if `None`.
        * `rng` <numpy.random.RandomState|None>
            A random number generator; not used in this class
        * `theano_rng` <theano.tensor.shared_randomstreams.RandomStreams|None>
            A symbolic random number generator; not used in this class
        """
        if getattr(n_in, "__len__", None) is not None:
            if len(n_in) > 1:
                n_in = (np.prod(n_in),)  # Flatten a multi-D input
        else:
            raise TypeError("The input `n_in` must be a tuple.")

        super(LCMaxoutLayer, self).__init__(n_in, **kwargs)
        self.n_units = n_units
        self.n_segments = n_segments
        self.n_out = (n_segments * n_units,)
        self.maxout_k = maxout_k
        self.activation = "maxout"
        self.activation_func = act.maxout

        ######################################
        # Initialize the trainable parameters.

        # Divide the input up into even-ish segments.
        self.input_segments = []
        seg_len = self.n_in[0] // self.n_segments
        self.n_in_segments = []
        for i_seg in range(self.n_segments):
            seg_start = i_seg * seg_len
            seg_end = seg_start + seg_len
            if i_seg == self.n_segments - 1:
                seg_end = self.n_in[0]  # Go to the end of the input with the last segment.
            self.n_in_segments.append(seg_end-seg_start)

        if W is None:
            W = []
            for i_seg, n_seg in enumerate(self.n_in_segments):
                maxout_group = []
                for i_maxout in range(self.maxout_k):
                    maxout_group.append(theano.shared(value=get_weight_init(
                        self.activation, (n_seg, n_units), n_seg, n_units, self.rng),
                                                      name="W_{}_{}_{}".format(self.name, i_seg, i_maxout),
                                                      borrow=True))
                W.append(maxout_group)
        if b is None:
            b = []
            for i_seg, n_seg in enumerate(self.n_in_segments):
                maxout_group = []
                for i_maxout in range(self.maxout_k):
                    b_values = np.zeros((n_units,), dtype=theano.config.floatX)
                    maxout_group.append(theano.shared(value=b_values,
                                                      name="b_{}_{}_{}".format(self.name, i_seg, i_maxout),
                                                      borrow=True))
                b.append(maxout_group)

        self.W = W
        self.b = b

        # Define regularization parameters.
        self.set_l1_l2_norms(self.W, self.l1_lambda, self.l2_lambda)

        # Store the parameters of the model.

        self.params = [maxout_w for w in self.W for maxout_w in w] + \
                      [maxout_b for b in self.b for maxout_b in b]

    def compile(self, input):
        input = input.flatten(2)  # Flatten a multi-D input
        super(LCMaxoutLayer, self).compile(input)

        output_segments = []
        seg_start = 0
        for i_seg, n_seg in enumerate(self.n_in_segments):
            self.input_segments.append(self.input[:, seg_start: seg_start + n_seg])
            seg_start += n_seg

            lin_output = [T.dot(self.input_segments[i_seg], W) + b
                          for W, b in zip(self.W[i_seg], self.b[i_seg])]
            output_segments.append(self.activation_func(lin_output))
        self.output = T.concatenate(output_segments, axis=1)

    def get_params(self):
        """Parameters required by __init__"""
        param_dict = super(LCMaxoutLayer, self).get_params()
        param_dict.update(n_units=self.n_units, n_segments=self.n_segments, maxout_k=self.maxout_k)
        return param_dict

    def get_trainable_params(self):
        """Returns a dictionary of parameters which can be inserted into this class's __init__.
        """
        return dict(W=self.W, b=self.b)

    def __str__(self):
        return "{}: Locally-connected flat layer with {} segments of {} neurons using {} " \
               "activation;\n\tLayer input shape = {}; output shape = {}; L1 penalty = {}, " \
               "L2 penalty = {}, max column norm = {}".\
            format(self.name, self.n_segments, self.n_units, self.activation, self.n_in, self.n_out,
                   self.l1_lambda, self.l2_lambda, self.maxnorm)


class FCLayer(Layer):
    def __init__(self, n_in, n_units, activation="prelu",
                 W=None, b=None, bvis=None, act_func_args=None, act_func_args_vis=None,
                 corruption_level=0.2, **kwargs):
        """This is a typical hidden layer of a multi-layer perceptron:
        units are fully-connected and have nonlinear activation functions.
        The weight matrix W is of shape (n_in, n_units) and the bias
        vector b is of shape (n_units,).

        Hidden unit activation will be given by: activation(dot(input, W) + b).

        **Parameters**

        * `n_in` <int>
            Dimensionality of input
        * `n_units` <int>
            Number of hidden units

        **Optional Parameters**

        * `name` <str|None>
            A designation for this layer. Can be used to retrieve it later.
        * `activation` <str|"prelu">
            The activation function to use for this layer's outputs.
        * `W` <theano.shared|None>
            Weights, shape (n_in, n_units); will be initialized randomly if `None`.
        * `b` <theano.shared|None>
            Biases, shape (n_units,); will be initialized to zero if `None`.
        * `bvis` <theano.shared|None>
            Biases of the visible layer when training as an autoencoder.
            Shape (n_units,); will be initialized to zero if `None`.
        * `act_func_args` <tuple of theano.shared|None>
            Arguments for activation functions with trainable parameters, e.g. the slope
            of a PReLU activation function.
        * `act_func_args_vis` <tuple of theano.shared|None>
            As the `act_func_args`, but for reconstructed visible units during autoencoder training.
        * `corruption_level` <float|0.2>
            Default corruption level used when training this layer as a denoising autoencoder.
        * `rng` <numpy.random.RandomState|None>
            A random number generator; not used in this class
        * `theano_rng` <theano.tensor.shared_randomstreams.RandomStreams|None>
            A symbolic random number generator; not used in this class
        """
        if getattr(n_in, "__len__", None) is not None:
            if len(n_in) > 1:
                n_in = (np.prod(n_in),)  # Flatten a multi-D input
        else:
            raise TypeError("The input `n_in` must be a tuple.")

        super(FCLayer, self).__init__(n_in, **kwargs)
        self.n_units = n_units
        self.n_out = (n_units,)
        self.activation = act.standardize_activation_name(activation)
        self.activation_func = act.get_activation_func(activation)  # Turn strings into functions.
        self.corruption_level = corruption_level

        if W is None:
            W = theano.shared(value=get_weight_init(self.activation, (n_in[0], n_units),
                                                    n_in[0], n_units, self.rng),
                              name="W_{}".format(self.name), borrow=True)
        if b is None:
            b_values = np.zeros((n_units,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name="b_{}".format(self.name), borrow=True)
        if act_func_args is None:
            act_func_args = init_act_func_args(self.activation, n_units, self.name)

        self.W = W
        self.b = b
        self.act_func_args = act_func_args

        # Use these parameters only if training as an autoencoder.
        # We'll initialize `bvis` to zeros only if we need it.
        self.bvis = bvis  # bvis corresponds to the bias of the visible units
        self.W_prime = self.W.T  # Use tied weights, therefore W_prime is W transpose.
        self.act_func_args_vis = act_func_args_vis  # Set only if autoencoder training requested.
        self.ac_params = []

        # Define regularization parameters.
        self.set_l1_l2_norms(self.W, self.l1_lambda, self.l2_lambda)

        # Store the parameters of the model.
        self.params = [self.W, self.b]
        self.params.extend(self.act_func_args)

    def compile(self, input):
        input = input.flatten(2)
        super(FCLayer, self).compile(input)  # Flatten a multi-D input

        self.output = self.activation_func(T.dot(self.input, self.W) + self.b, *self.act_func_args)

    def _get_corrupted_input(self, input, corruption_level):
        """This function keeps `1-corruption_level` entries of the inputs the
        same and zero-out randomly selected subset of size `coruption_level`
        Note : The first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
            The second argument is the number of trials
            The third argument is the probability of success of any trial

            This will produce an array of 0s and 1s where 1 has a
                probability of 1 - `corruption_level` and 0 with
                `corruption_level`

                The binomial function returns the int64 data type by
                default.  int64 multiplied by the input
                type(floatX) always returns float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, so this doesn't change the
                result. This is needed to allow the GPU to work
                correctly as it only supports float32 for now.

        This function exists to support training this class as a denoising autoencoder.
        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def _get_hidden_values(self, input):
        """ Computes the values of the hidden layer. """
        return self.activation_func(T.dot(input, self.W) + self.b, *self.act_func_args)

    def _get_reconstructed_input(self, hidden):
        """Computes the reconstructed visible values given the values of the hidden layer.
        """
        return self.activation_func(T.dot(hidden, self.W_prime) + self.bvis, *self.act_func_args_vis)

    def get_autoencoder_loss(self, corruption_level, regularized=False, loss="squared"):
        """ This function computes the cost and the updates for one training
        step of the dA. """
        # First initialize the autoencoder-only trainable parameters.
        if self.bvis is None:
            # We'll only initialize this if we need it.
            bvis_values = np.zeros((self.n_in[0],), dtype=theano.config.floatX)
            self.bvis = theano.shared(value=bvis_values, name="bvis_{}".format(self.name), borrow=True)
        if not self.act_func_args_vis:
            self.act_func_args_vis = init_act_func_args(self.activation, self.n_in[0],
                                                        "vis_{}".format(self.name))
        if not self.ac_params:
            self.ac_params = self.params.copy() + [self.bvis]
            self.ac_params.extend(self.act_func_args_vis)

        tilde_x = self._get_corrupted_input(self.input, corruption_level)
        y = self._get_hidden_values(tilde_x)
        z = self._get_reconstructed_input(y)

        # Note : We sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        if loss == "cross-entropy":
            minibatch_loss = - T.sum(self.input * T.log(z) + (1 - self.input) * T.log(1 - z), axis=1)
        elif loss == "squared":
            minibatch_loss = T.sum((self.input - z) ** 2, axis=1)
        else:
            raise ValueError("Unrecognized loss function: \"{}\".".format(loss))

        # Note : `minibatch_loss` is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch.
        loss = T.mean(minibatch_loss)
        if regularized:
            loss = loss + self.l1 + self.l2_sqr

        return loss

    def get_params(self):
        """Parameters required by __init__"""
        param_dict = super(FCLayer, self).get_params()
        param_dict.update(n_units=self.n_units, activation=self.activation)
        return param_dict

    def get_trainable_params(self):
        """Returns a dictionary of parameters which can be inserted into this class's __init__.
        """
        return dict(W=self.W, b=self.b, bvis=self.bvis,
                    act_func_args=self.act_func_args, act_func_args_vis=self.act_func_args_vis)

    def __str__(self):
        return "{}: Fully-connected layer with {} neurons using {} activation;\n\t" \
               "FC layer input shape = {}; output shape = {}; L1 penalty = {}, " \
               "L2 penalty = {}, max column norm = {}".\
            format(self.name, self.n_units, self.activation, self.n_in, self.n_out,
                   self.l1_lambda, self.l2_lambda, self.maxnorm)


class FCMaxoutLayer(Layer):
    """
    This is a typical hidden layer of a multi-layer perceptron:
    units are fully-connected and have nonlinear activation functions.
    The weight matrix W is of shape (n_in, n_units) and the bias
    vector b is of shape (n_units,).

    In this case, the activation function is "maxout", as described in

    "Maxout Networks". Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
    Courville, and Yoshua Bengio. ICML 2013. arXiv:1302.4389 (http://arxiv.org/abs/1302.4389).
    """
    def __init__(self, n_in, n_units, maxout_k=5,
                 W=None, b=None, **kwargs):
        """Initialization

        **Parameters**

        * `n_in` <int>
            Dimensionality of input
        * `n_units` <int>
            Number of hidden units

        **Optional Parameters**

        * `name` <str|None>
            A designation for this layer. Can be used to retrieve it later.
        * `maxout_k` <int|5>
            Number of sets of weights and biases to create for the maxout.
        * `W` <theano.shared|None>
            Weights, shape (n_in, n_units); will be initialized randomly if `None`.
        * `b` <theano.shared|None>
            Biases, shape (n_units,); will be initialized to zero if `None`.
        * `rng` <numpy.random.RandomState|None>
            A random number generator; not used in this class
        * `theano_rng` <theano.tensor.shared_randomstreams.RandomStreams|None>
            A symbolic random number generator; not used in this class
        """
        if getattr(n_in, "__len__", None) is not None:
            if len(n_in) > 1:
                # Flatten a multi-D input
                n_in = (np.prod(n_in),)
        else:
            raise TypeError("The input `n_in` must be a tuple.")

        super(FCMaxoutLayer, self).__init__(n_in, **kwargs)
        self.n_units = n_units
        self.n_out = (n_units,)
        self.maxout_k = maxout_k
        self.activation = "maxout"
        self.activation_func = act.maxout

        if W is None:
            W = []
            for i_w in range(self.maxout_k):
                W.append(theano.shared(value=get_weight_init("maxout", (n_in[0], n_units),
                                                             n_in[0], n_units, self.rng),
                                       name="W_{}_{}".format(self.name, i_w), borrow=True))
        else:
            if len(W) != self.maxout_k:
                raise ValueError("Please input weights as a list of "
                                 "{} weights.".format(self.maxout_k))
        if b is None:
            b = []
            for i_b in range(self.maxout_k):
                # The bias is a 1D tensor -- one bias per output feature map.
                b_values = np.zeros((self.n_units,), dtype=theano.config.floatX)
                b.append(theano.shared(value=b_values, name="b_{}_{}".format(self.name, i_b), borrow=True))
        else:
            if len(b) != self.maxout_k:
                raise ValueError("Please input biases as a list of "
                                 "{} biases.".format(self.maxout_k))
        self.W = W
        self.b = b

        # Define regularization parameters.
        self.l1, self.l2_sqr = None, None
        if self.l1_lambda:
            self.l1 = self.l1_lambda * np.sum([np.abs(this_w) for this_w in self.W]).sum()
        if self.l2_lambda:
            self.l2_sqr = self.l2_lambda * np.sum([this_w ** 2 for this_w in self.W]).sum()

        # Store the parameters of the model.
        self.params = self.W + self.b

    def compile(self, input):
        input = input.flatten(2)
        super(FCMaxoutLayer, self).compile(input)

        lin_output = [(T.dot(self.input, this_w) + this_b)
                      for this_w, this_b in zip(self.W, self.b)]
        self.output = self.activation_func(lin_output)

    def get_params(self):
        """Parameters required by __init__"""
        param_dict = super(FCMaxoutLayer, self).get_params()
        param_dict.update(n_units=self.n_units, maxout_k=self.maxout_k)
        return param_dict

    def get_trainable_params(self):
        """Returns a dictionary of parameters which can be inserted into this class's __init__.
        """
        return dict(W=self.W, b=self.b)

    def __str__(self):
        return "{}: Fully-connected layer with {} neurons using maxout activation (k={});\n\t" \
               "FC layer input shape = {}; output shape = {}; L1 penalty = {}, " \
               "L2 penalty = {}, max column norm = {}".\
            format(self.name, self.n_units, self.maxout_k, self.n_in, self.n_out,
                   self.l1_lambda, self.l2_lambda, self.maxnorm)

class DenoisingAutoEncoderLayer(FCLayer):
    """Denoising Auto-Encoder class (dA)

    A denoising autoencoders tries to reconstruct the input from a corrupted
    version of it by projecting it first in a latent space and reprojecting
    it afterwards back in the input space. Please refer to Vincent et al.,2008
    for more details. If x is the input then equation (1) computes a partially
    destroyed version of x by means of a stochastic mapping q_D. Equation (2)
    computes the projection of the input into the latent space. Equation (3)
    computes the reconstruction of the input, while equation (4) computes the
    reconstruction error.

    .. math::

        \tilde{x} ~ q_D(\tilde{x}|x)                                     (1)

        y = s(W \tilde{x} + b)                                           (2)

        x = s(W' y  + b')                                                (3)

        L(x,z) = -sum_{k=1}^d [x_k \log z_k + (1-x_k) \log( 1-z_k)]      (4)

    """
    def __init__(self, n_in, n_units, activation="prelu",
                 W=None, b=None, bvis=None, a=None, **kwargs):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        super(DenoisingAutoEncoderLayer, self).__init__(n_in, n_units, activation=activation,
                                                        W=W, b=b, a=a, **kwargs)

        if bvis is None:
            bvis_values = np.zeros((n_units,), dtype=theano.config.floatX)
            bvis = theano.shared(value=bvis_values, name="bvis_{}".format(self.name), borrow=True)

        # b_prime corresponds to the bias of the visible units
        self.b_prime = bvis
        # Use tied weights, therefore W_prime is W transpose.
        self.W_prime = self.W.T

        self.params.append(self.b_prime)

    def get_corrupted_input(self, input, corruption_level):
        """This function keeps `1-corruption_level` entries of the inputs the
        same and zero-out randomly selected subset of size `coruption_level`
        Note : The first argument of theano.rng.binomial is the shape(size) of
               random numbers that it should produce
            The second argument is the number of trials
            The third argument is the probability of success of any trial

            This will produce an array of 0s and 1s where 1 has a
                probability of 1 - `corruption_level` and 0 with
                `corruption_level`

                The binomial function returns the int64 data type by
                default.  int64 multiplied by the input
                type(floatX) always returns float64.  To keep all data
                in floatX when floatX is float32, we set the dtype of
                the binomial to floatX. As in our case the value of
                the binomial is always 0 or 1, so this doesn't change the
                result. This is needed to allow the GPU to work
                correctly as it only supports float32 for now.

        """
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer. """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed visible values given the values of the hidden layer.
        """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_autoencoder_loss(self, corruption_level, regularized=False):
        """ This function computes the cost and the updates for one training
        step of the dA. """

        tilde_x = self.get_corrupted_input(self.input, corruption_level)
        y = self.get_hidden_values(tilde_x)
        z = self.get_reconstructed_input(y)

        # Note : We sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        minibatch_loss = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)

        # Note : `minibatch_loss` is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch.
        loss = T.mean(minibatch_loss)
        if regularized:
            loss = loss + self.l1 + self.l2_sqr

        return loss

    def get_trainable_params(self):
        """Returns a dictionary of parameters which can be inserted into this class's __init__.
        """
        return dict(W=self.W, b=self.b, bvis=self.b_prime, a=self.a)

    def __str__(self):
        return "{}: Fully-connected (dA) layer with {} neurons using {} activation;\n\t" \
               "FC layer input shape = {}; output shape = {}; L1 penalty = {}, " \
               "L2 penalty = {}, max column norm = {}".\
            format(self.name, self.n_units, self.activation, self.n_in, self.n_out,
                   self.l1_lambda, self.l2_lambda, self.maxnorm)


class ConvLayer(Layer):
    """Convolutional layer"""
    def __init__(self, n_in, n_output_maps, filter_shape, activation="prelu", stride=(1, 1),
                 W=None, b=None, act_func_args=None, batch_size=None, partial_sum=1, **kwargs):
        """Allocate a convolutional layer with shared variable internal parameters.

        * `n_in` <tuple or list of length 4>
            Shape of the input: (num input feature maps, image height, image width)
        * `n_output_maps` <int>
            Number of output kernels (or maps)
        * `filter_shape` <tuple or list of length 2>
            Size of the convolution region, in pixels.

        **Optional Parameters**

        * `name` <str|None>
            A designation for this layer. Can be used to retrieve it later.
        * `stride` <tuple|(1, 1)>
            Take steps of this size across the image.
        * `activation` <str|"prelu">
            The activation function to use for this layer's outputs.
        * `W` <theano.shared|None>
            Weights, shape (n_in, n_units); will be initialized randomly if `None`.
        * `b` <theano.shared|None>
            Biases, shape (n_units,); will be initialized to zero if `None`.
        * `act_func_args` <tuple of theano.shared|None>
            Arguments for activation functions with trainable parameters, e.g. the slope
            of a PReLU activation function.
        * `batch_size` <int|None>
            If None, allow for arbitrary batch sizes. Otherwise, optimize the
            convolution operation for minibatches of size `batch_size`.
        * `partial_sum` <int|1>
            Optimization parameter for cuda-convnet. Higher values improve speed and increase
            memory use. From cuda-convnet docs: "Valid values are ones that divide the area of the
            output grid in this convolutional layer. For example if this layer produces
            32-channel 20x20 output grid, valid values of partialSum are ones which
            divide 20*20 = 400."
        * `rng` <numpy.random.RandomState|None>
            A random number generator; not used in this class
        * `theano_rng` <theano.tensor.shared_randomstreams.RandomStreams|None>
            A symbolic random number generator; not used in this class
        """
        super(ConvLayer, self).__init__(n_in, **kwargs)
        self.n_output_maps = n_output_maps
        self.filter_shape = filter_shape
        self.batch_size = batch_size
        self.activation = act.standardize_activation_name(activation)
        self.activation_func = act.get_activation_func(activation)  # Turn strings into functions.
        self.stride = stride
        self.partial_sum = partial_sum

        # Type-check the inputs.
        if getattr(n_in, "__len__", lambda: 0)() != 3:
            raise TypeError("The input `n_in` must be a 3-tuple: "
                            "(num input feature maps, image height, image width).")
        if getattr(filter_shape, "__len__", lambda: 0)() != 2:
            raise TypeError("Please specify a width and a height for the convolutional filter.")
        if not isinstance(n_output_maps, int):
            raise TypeError("The `n_output_maps` input must be a single integer.")
        if isinstance(stride, int):
            self.stride = (stride, stride)

        self.n_out = (self.n_output_maps,
                      int(np.ceil((self.n_in[1] - self.filter_shape[0]) / self.stride[0])) + 1,
                      int(np.ceil((self.n_in[2] - self.filter_shape[1]) / self.stride[1])) + 1)

        #optimize_with_cuda_convnet = CUDA_CONVNET and batch_size is not None
        if CUDA_CONVNET:
            # Make sure that the inputs are valid for cuda-convnet.
            if self.filter_shape[0] != self.filter_shape[1]:
                raise ValueError("Cuda-convnet requires square filters.")
            if self.stride[0] != self.stride[1]:
                raise ValueError("Cuda-convnet requires equal strides in each dimension.")
            if n_in[0] % 4 != 0 and n_in[0] > 4:
                raise ValueError("Cuda-convnet requires a number of input "
                                 "channels divisible by or less than 4.")
            if self.n_output_maps % 16 != 0:
                raise ValueError("Cuda-convnet requires that the number of filters "
                                 "be a multiple of 16.")
            if self.n_out[1] * self.n_out[2] % self.partial_sum != 0:
                raise ValueError("The partial sum must divide the number of image pixels "
                                 "({}).".format(self.n_out[1] * self.n_out[2]))
            if self.batch_size is not None and self.batch_size % 128 != 0:
                log.warning("Use batch size a multiple of 128 for optimal performance.")
        elif batch_size is None:
            log.warning("`batch_size` is None. Please specify batch size for "
                        "improved training speed.")

        # There are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit.
        fan_in = n_in[0] * np.prod(filter_shape)

        # Each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width"
        fan_out = self.n_output_maps * np.prod(filter_shape)

        # Initialize weights with random weights.
        if CUDA_CONVNET:
            # Weights in c01b shape.
            self.conv_filter_shape = [self.n_in[0], self.filter_shape[0],
                                      self.filter_shape[1], self.n_output_maps]
        else:
            # Weights in bc01 shape.
            self.conv_filter_shape = [self.n_output_maps, self.n_in[0],
                                      self.filter_shape[0], self.filter_shape[1]]
        if W is None:
            n_hat = fan_out if activation == act.parametric_rectifier else fan_in
            W = theano.shared(value=get_weight_init(self.activation, self.conv_filter_shape,
                                                    n_hat, fan_out, self.rng),
                              name="W_{}".format(self.name), borrow=True)
        if b is None:
            # The bias is a 1D tensor -- one bias per output feature map.
            b_values = np.zeros((self.n_output_maps,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name="b_{}".format(self.name), borrow=True)
        if act_func_args is None:
            act_func_args = init_act_func_args(self.activation, self.n_output_maps, self.name)

        self.W = W
        self.b = b #.dimshuffle(0, "x", "x", "x") if CUDA_CONVNET else b.dimshuffle("x", 0, "x", "x")
        # Make sure the activation function arguments have the right shape.

        self.act_func_args = act_func_args
        # if CUDA_CONVNET:
        #     self.act_func_args = [a.dimshuffle(0, "x", "x", "x") for a in act_func_args]
        # else:
        #     self.act_func_args = [a.dimshuffle("x", 0, "x", "x") for a in act_func_args]

        # Define regularization parameters.
        self.set_l1_l2_norms(self.W, self.l1_lambda, self.l2_lambda)

        # Store the parameters of the model.
        self.params = [self.W, self.b]
        self.params.extend(self.act_func_args)

    def compile(self, input):
        super(ConvLayer, self).compile(input)

        # Convolve input feature maps with filters.
        # Note: If we don't specify the `filter_shape` and `image_shape`, this function
        # will worth with arbitrary input shapes -- i.e., you don't have to pre-set the batch size.
        # However, this means that Theano won't be able to optimize the operation.
        # I've observed that complex CNN models run more than twice as slowly when
        # I don't specify `filter_shape` and `image_shape` here!
        if CUDA_CONVNET:
            # Use code snippets from
            # http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
            # to achieve faster convolutions. Note that the cuda-convnet library
            # assumes a different ordering for the inputs than Theano, so we need to
            # shuffle arrays back and forth.
            conv_op = filter_acts.FilterActs(partial_sum=self.partial_sum, stride=self.stride[0])
            contiguous_input = gpu_contiguous(self.input)
            contiguous_filters = gpu_contiguous(self.W)

            conv_out_inf = conv_out = conv_op(contiguous_input, contiguous_filters)
            dimshuffle_args = (0, "x", "x", "x")
            #bias_shuffle = self.b.dimshuffle(0, "x", "x", "x")
        else:
            conv_out = conv.conv2d(input=self.input, filters=self.W, subsample=self.stride,
                                   filter_shape=(self.conv_filter_shape
                                                 if self.batch_size else None),
                                   image_shape=((self.batch_size,) + self.n_in
                                                if self.batch_size else None))
            #bias_shuffle = self.b.dimshuffle("x", 0, "x", "x")

            # This should be slower than specifying image shapes, but it will work with
            # arbitrary input sizes, so it can be helpful for predicting.
            conv_out_inf = conv.conv2d(self.input, filters=self.W, subsample=self.stride)
            dimshuffle_args = ("x", 0, "x", "x")

        # Add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map width & height.
        self.output = self.activation_func(conv_out + self.b.dimshuffle(*dimshuffle_args),
                                           *[a.dimshuffle(*dimshuffle_args) for a in self.act_func_args])
        self.output_inf = self.activation_func(conv_out_inf + self.b.dimshuffle(*dimshuffle_args),
                                               *[a.dimshuffle(*dimshuffle_args) for a in self.act_func_args])

    def get_params(self):
        """Parameters required by __init__"""
        param_dict = super(ConvLayer, self).get_params()
        param_dict.update(n_output_maps=self.n_output_maps, filter_shape=self.filter_shape,
                          stride=self.stride, activation=self.activation,
                          partial_sum=self.partial_sum, batch_size=self.batch_size)
        return param_dict

    def get_trainable_params(self):
        """Returns a dictionary of parameters which can be inserted into this class's __init__.
        """
        return dict(W=self.W, b=self.b, act_func_args=self.act_func_args)

    def __str__(self):
        return "{}: Convolutional layer with filter shape {}, stride {}, and {} output kernels, " \
               "using {} activation;\n\tConv layer input shape = {}; output shape = {}; " \
               "L1 penalty = {}, L2 penalty = {}, max column norm = {}".\
            format(self.name, self.filter_shape, self.stride, self.n_output_maps, self.activation,
                   self.n_in, self.n_out, self.l1_lambda, self.l2_lambda, self.maxnorm)


class ConvMaxoutLayer(Layer):
    """Convolutional layer that uses the Maxout activation function, as
    described in

    "Maxout Networks". Ian J. Goodfellow, David Warde-Farley, Mehdi Mirza, Aaron
    Courville, and Yoshua Bengio. ICML 2013. arXiv:1302.4389 (http://arxiv.org/abs/1302.4389).
    """
    def __init__(self, n_in, n_output_maps, filter_shape, maxout_k=5,
                 stride=(1, 1), W=None, b=None, batch_size=None, partial_sum=1, **kwargs):
        """Allocate a convolutional layer with shared variable internal parameters.

        * `n_in` <tuple or list of length 4>
            Shape of the input: (num input feature maps, image height, image width)
        * `n_output_maps` <int>
            Number of output kernels (or maps)
        * `filter_shape` <tuple or list of length 2>
            Size of the convolution region, in pixels.

        **Optional Parameters**

        * `stride` <tuple|(1, 1)>
            Take steps of this size across the image.
        * `maxout_k` <int|5>
            Number of sets of weights and biases to create for the maxout.
        * `W` <theano.shared|None>
            Weights, list of `maxout_k` entries with shape (n_in, n_units);
            will be initialized randomly if `None`.
        * `b` <theano.shared|None>
            Biases, list of `maxout_k` entries with shape (n_units,);
            will be initialized to zero if `None`.
        * `batch_size` <int|None>
            If None, allow for arbitrary batch sizes. Otherwise, optimize the
            convolution operation for minibatches of size `batch_size`.
        * `partial_sum` <int|1>
            Optimization parameter for cuda-convnet. Higher values improve speed and increase
            memory use. From cuda-convnet docs: "Valid values are ones that divide the area of the
            output grid in this convolutional layer. For example if this layer produces
            32-channel 20x20 output grid, valid values of partialSum are ones which
            divide 20*20 = 400."
        * `rng` <numpy.random.RandomState|None>
            A random number generator; not used in this class
        * `theano_rng` <theano.tensor.shared_randomstreams.RandomStreams|None>
            A symbolic random number generator; not used in this class
        """
        super(ConvMaxoutLayer, self).__init__(n_in, **kwargs)
        self.n_output_maps = n_output_maps
        self.filter_shape = filter_shape
        self.batch_size = batch_size
        self.maxout_k = maxout_k
        self.stride = stride
        self.partial_sum = partial_sum

        # Type-check the inputs.
        if getattr(n_in, "__len__", lambda: 0)() != 3:
            raise TypeError("The input `n_in` must be a 3-tuple: "
                            "(num input feature maps, image height, image width).")
        if getattr(filter_shape, "__len__", lambda: 0)() != 2:
            raise TypeError("Please specify a width and a height for the convolutional filter.")
        if not isinstance(n_output_maps, int):
            raise TypeError("The `n_output_maps` input must be a single integer.")
        if isinstance(stride, int):
            self.stride = (stride, stride)

        #optimize_with_cuda_convnet = CUDA_CONVNET and batch_size is not None
        if CUDA_CONVNET:
            # Make sure that the inputs are valid for cuda-convnet.
            if self.filter_shape[0] != self.filter_shape[1]:
                raise ValueError("Cuda-convnet requires square filters.")
            if self.stride[0] != self.stride[1]:
                raise ValueError("Cuda-convnet requires equal strides in each dimension.")
            if n_in[0] % 4 != 0 and n_in[0] > 4:
                raise ValueError("Cuda-convnet requires a number of input "
                                 "channels divisible by or less than 4.")
            if self.n_output_maps % 16 != 0:
                raise ValueError("Cuda-convnet requires that the number of filters "
                                 "be a multiple of 16.")
            if self.batch_size is not None and self.batch_size % 128 != 0:
                log.warning("Use batch size a multiple of 128 for optimal performance.")
        elif self.batch_size is None:
            log.warning("`batch_size` is None. Please specify batch size for "
                            "improved training speed.")

        self.n_out = (self.n_output_maps,
                      int(np.ceil((self.n_in[1] - self.filter_shape[0]) / self.stride[0])) + 1,
                      int(np.ceil((self.n_in[2] - self.filter_shape[1]) / self.stride[1])) + 1)

        # There are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit.
        fan_in = n_in[0] * np.prod(filter_shape)

        # Each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width"
        fan_out = self.n_output_maps * np.prod(filter_shape)

        # Initialize weights with random weights..
        if CUDA_CONVNET:
            # Weights in c01b shape.
            self.conv_filter_shape = [self.n_in[0], self.filter_shape[0],
                                      self.filter_shape[1], self.n_output_maps]
        else:
            # Weights in bc01 shape.
            self.conv_filter_shape = [self.n_output_maps, self.n_in[0],
                                      self.filter_shape[0], self.filter_shape[1]]
        if W is None:
            n_hat = fan_out
            W = []
            for i_w in range(self.maxout_k):
                W.append(theano.shared(value=get_weight_init("maxout", self.conv_filter_shape,
                                                             n_hat, fan_out, self.rng),
                                       name="W_{}_{}".format(self.name, i_w), borrow=True))
        else:
            if len(W) != self.maxout_k:
                raise ValueError("Please input weights as a list of "
                                 "{} weights.".format(self.maxout_k))
        if b is None:
            b = []
            for i_b in range(self.maxout_k):
                # The bias is a 1D tensor -- one bias per output feature map.
                b_values = np.zeros((self.n_output_maps,), dtype=theano.config.floatX)
                b.append(theano.shared(value=b_values, borrow=True,
                                       name="b_{}_{}".format(self.name, i_b)))
        else:
            if len(b) != self.maxout_k:
                raise ValueError("Please input biases as a list of "
                                 "{} biases.".format(self.maxout_k))
        self.W = W
        self.b = b

        # Define regularization parameters.
        self.l1, self.l2_sqr = None, None
        if self.l1_lambda:
            self.l1 = self.l1_lambda * np.sum([np.abs(this_w) for this_w in self.W]).sum()
        if self.l2_lambda:
            self.l2_sqr = self.l2_lambda * np.sum([this_w ** 2 for this_w in self.W]).sum()

        # Store the parameters of the model.
        self.params = self.W + self.b

    def compile(self, input):
        super(ConvMaxoutLayer, self).compile(input)

        # Convolve input feature maps with filters.
        # Note: If we don't specify the `filter_shape` and `image_shape`, this function
        # will worth with arbitrary input shapes -- i.e., you don't have to pre-set the batch size.
        # However, this means that Theano won't be able to optimize the operation.
        # I've observed that complex CNN models run more than twice as slowly when
        # I don't specify `filter_shape` and `image_shape` here!
        # For this maxout implementation, find the list of "z" parameters individually
        # for each linear activation, then combine them with the call to `maxout`.
        z = []
        z_inf = []
        for this_w, this_b in zip(self.W, self.b):
            if CUDA_CONVNET:
                # Use code snippets from
                # http://benanne.github.io/2014/04/03/faster-convolutions-in-theano.html
                # to achieve faster convolutions. Note that the cuda-convnet library
                # assumes a different ordering for the inputs than Theano, so we need to
                # shuffle arrays back and forth.
                conv_op = filter_acts.FilterActs(partial_sum=self.partial_sum, stride=self.stride[0])
                contiguous_input = gpu_contiguous(self.input)
                contiguous_filters = gpu_contiguous(this_w)

                conv_inf = conv_out = conv_op(contiguous_input, contiguous_filters)
                dimshuffle_args = (0, "x", "x", "x")
                #bias_shuffle = this_b.dimshuffle(0, "x", "x", "x")
            else:
                conv_out = conv.conv2d(input=self.input, filters=this_w, subsample=self.stride,
                                       filter_shape=(self.conv_filter_shape
                                                     if self.batch_size else None),
                                       image_shape=((self.batch_size,) + self.n_in
                                                    if self.batch_size else None))
                #bias_shuffle = this_b.dimshuffle("x", 0, "x", "x")

                # This should be slower than specifying image shapes, but it will work with
                # arbitrary input sizes, so it can be helpful for predicting.
                conv_inf = conv.conv2d(input=self.input, filters=this_w, subsample=self.stride)
                dimshuffle_args = ("x", 0, "x", "x")
            z.append(conv_out + this_b.dimshuffle(*dimshuffle_args))
            z_inf.append(conv_inf + this_b.dimshuffle(*dimshuffle_args))

        self.output = act.maxout(z)
        self.output_inf = act.maxout(z_inf)

    def get_params(self):
        """Parameters required by __init__"""
        param_dict = super(ConvMaxoutLayer, self).get_params()
        param_dict.update(n_output_maps=self.n_output_maps, filter_shape=self.filter_shape,
                          partial_sum=self.partial_sum, stride=self.stride,
                          maxout_k=self.maxout_k, batch_size=self.batch_size)
        return param_dict

    def get_trainable_params(self):
        """Returns a dictionary of parameters which can be inserted into this class's __init__.
        """
        return dict(W=self.W, b=self.b)

    def __str__(self):
        return "{}: Convolutional layer with filter shape {}, stride {}, and {} output kernels, using " \
               "maxout (k={}) activation;\n\tConv layer input shape = {}; output shape = {}; " \
               "L1 penalty = {}, L2 penalty = {}, max column norm = {}".\
            format(self.name, self.filter_shape, self.stride, self.n_output_maps, self.maxout_k,
                   self.n_in, self.n_out, self.l1_lambda, self.l2_lambda, self.maxnorm)


class MLPConvLayer(Layer):
    """This class implements an "mlpconv" layer as described in
    M. Lin, Q. Chen, and S. Yan, "Network In Network", arXiv:1312.4400v3

    It's best described as a convolutional layer with a multi-layer perceptron
    as the "activation function". In this implementation, the MLP is implemented as a series of
    convolutions with a 1x1 filter.
    """
    def __init__(self, n_in, n_output_maps, filter_shape, n_units, activation="prelu",
                 stride=(1, 1), W=None, b=None, act_func_args=None, batch_size=None,
                 partial_sum=1, **kwargs):
        """Allocate a convolutional layer with shared variable internal parameters.

        * `n_in` <tuple or list of length 4>
            Shape of the input: (num input feature maps, image height, image width)
        * `n_output_maps` <int>
            Number of output kernels (or maps)
        * `filter_shape` <tuple or list of length 2>
            Size of the convolution region, in pixels.
        * `n_units` <list of int>
            Number of hidden units in each internal layer of the MLPConvLayer.

        **Optional Parameters**

        * `name` <str|None>
            A designation for this layer. Can be used to retrieve it later.
        * `activation` <str|"prelu">
            The activation function to use for each of this layer's hidden layer outputs.
        * `stride` <tuple|(1, 1)>
            Take steps of this size across the image.
        * `W` <list of theano.shared|None>
            Weights, shape (n_in, n_units); will be initialized randomly if `None`.
            Each element in the list of weight arrays corresponds to one hidden layer's weights.
        * `b` <list of theano.shared|None>
            Biases, shape (n_units,); will be initialized to zero if `None`.
        * `act_func_args` <list of tuples of theano.shared|None>
            Arguments for activation functions with trainable parameters, e.g. the slope
            of a PReLU activation function.
        * `batch_size` <int|None>
            If None, allow for arbitrary batch sizes. Otherwise, optimize the
            convolution operation for minibatches of size `batch_size`.
        * `partial_sum` <int|1>
            Optimization parameter for cuda-convnet. Higher values improve speed and increase
            memory use. From cuda-convnet docs: "Valid values are ones that divide the area of the
            output grid in this convolutional layer. For example if this layer produces
            32-channel 20x20 output grid, valid values of partialSum are ones which
            divide 20*20 = 400."
        * `rng` <numpy.random.RandomState|None>
            A random number generator; not used in this class
        * `theano_rng` <theano.tensor.shared_randomstreams.RandomStreams|None>
            A symbolic random number generator; not used in this class
        """
        super(MLPConvLayer, self).__init__(n_in, **kwargs)
        self.n_output_maps = n_output_maps
        self.filter_shape = filter_shape
        self.n_units = n_units
        self.batch_size = batch_size
        self.activation = activation
        self.stride = stride
        self.partial_sum = partial_sum

        # Type-check the inputs.
        if getattr(n_in, "__len__", lambda: 0)() != 3:
            raise TypeError("The input `n_in` must be a 3-tuple: "
                            "(num input feature maps, image height, image width).")
        if getattr(filter_shape, "__len__", lambda: 0)() != 2:
            raise TypeError("Please specify a width and a height for the convolutional filter.")
        if not isinstance(n_output_maps, int):
            raise TypeError("The `n_output_maps` input must be a single integer.")
        if batch_size is None:
            log.warning("`batch_size` is None. Please specify batch size for "
                            "improved training speed.")
        if isinstance(stride, int):
            self.stride = (stride, stride)

        self.n_out = (self.n_output_maps,
                      int(np.ceil((self.n_in[1] - self.filter_shape[0]) / self.stride[0])) + 1,
                      int(np.ceil((self.n_in[2] - self.filter_shape[1]) / self.stride[1])) + 1)

        # Enforce that each parameter input must have len(n_units) + 1 : One set of parameters
        # for each of the internal layers, and one set for the first convolution.
        if W is None:
            W = (len(self.n_units) + 1) * [None]
        elif len(W) != len(self.n_units) + 1:
            raise ValueError("An input list of weights must have length = len(n_units)+1.")
        if b is None:
            b = (len(self.n_units) + 1) * [None]
        elif len(b) != len(self.n_units) + 1:
            raise ValueError("An input list of biases must have length = len(n_units)+1.")
        if act_func_args is None:
            act_func_args = (len(self.n_units) + 1) * [None]
        elif len(act_func_args) != len(self.n_units) + 1:
            raise ValueError("An input list of activation function parameters must "
                             "have length = len(n_units) + 1.")

        # Set up the internal layers of the mlpconv. Each is a 1x1 convolution.
        prev_n_out = n_in
        self.hidden_layers = []
        filters = [filter_shape] + (len(n_units) * [(1, 1)])
        self.W, self.b, self.act_func_args = [], [], []
        self.params = []
        maps_per_layer = self.n_units + [self.n_output_maps]
        for i_lyr, (n, filt, layer_w, layer_b, layer_a) in \
              enumerate(zip(maps_per_layer, filters, W, b, act_func_args)):
            lyr_kwargs = kwargs.copy()
            lyr_kwargs["name"] = kwargs["name"] + "_internal{}".format(i_lyr)
            if i_lyr == 0:
                lyr_kwargs["stride"] = self.stride  # Internal layers have default (1, 1) stride.
            layer = ConvLayer(n_in=prev_n_out, n_output_maps=n,
                              filter_shape=filt, activation=activation,
                              W=layer_w, b=layer_b, act_func_args=layer_a,
                              batch_size=batch_size, partial_sum=partial_sum, **lyr_kwargs)
            layer_params = layer.get_trainable_params()
            self.W.append(layer_params["W"])
            self.b.append(layer_params["b"])
            self.act_func_args.append(layer_params["act_func_args"])
            self.params.extend(layer.params)
            prev_n_out = layer.n_out

            self.hidden_layers.append(layer)
            log.debug("Added internal layer: {}".format(str(layer)))

    def compile(self, input):
        super(MLPConvLayer, self).compile(input)

        # Compile each of the internal layers.
        prev_output = self.input
        for layer in self.hidden_layers:
            layer.compile(prev_output)
            prev_output = layer.output

        self.output = self.hidden_layers[-1].output

    def get_params(self):
        """Parameters required by __init__"""
        param_dict = super(MLPConvLayer, self).get_params()
        param_dict.update(n_output_maps=self.n_output_maps, filter_shape=self.filter_shape,
                          activation=self.activation, batch_size=self.batch_size,
                          stride=self.stride, partial_sum=self.partial_sum, n_units=self.n_units)
        return param_dict

    def get_trainable_params(self):
        """Returns a dictionary of parameters which can be inserted into this class's __init__.
        """
        return dict(W=self.W, b=self.b, act_func_args=self.act_func_args)

    def __str__(self):
        return "{}: MLPConv layer with filter shape {} and {} output kernels;\n\tUsing " \
               "{} activation for internal layers {};\n\tConv layer input " \
               "shape = {}; output shape = {}; " \
               "L1 penalty = {}, L2 penalty = {}, max column norm = {}".\
            format(self.name, self.filter_shape, self.n_output_maps, self.activation, self.n_units,
                   self.n_in, self.n_out, self.l1_lambda, self.l2_lambda, self.maxnorm)


class GlobalAveragePoolLayer(PassthroughLayer):
    """This class implements the global average pooling described in
    M. Lin, Q. Chen, and S. Yan, "Network In Network".

    In that paper, the authors recommend connecting a series of MLPConv layers, finishing
    with a number of output feature maps equal to the number of classes for the classification
    task. Then one applies global average pooling, taking the average of each map.
    The resulting n_classes maps are fed directly into a softmax output layer.
    """
    def __init__(self, n_in, **kwargs):
        if len(n_in) != 3:
            raise TypeError("This layer requires a batch of 3D inputs.")
        super(GlobalAveragePoolLayer, self).__init__(n_in, **kwargs)

        self.n_out = self.n_in[:-2]

    def compile(self, input):
        super(GlobalAveragePoolLayer, self).compile(input)
        self.output = T.mean(self.input, axis=[2, 3], dtype=theano.config.floatX, keepdims=False)


class MaxPoolChannelsLayer(PassthroughLayer):
    """Over each group of k feature maps, pick the maximum value at each pixel.
    This layer does not reduce the size of maps, as the `MaxPool2DLayer` does,
    but does reduce the total number of maps.

    This max pool layer is used as the "activation function" of a Maxout layer.
    "Maxout Networks" Ian J. Goodfellow, David Warde-Farley,
    Mehdi Mirza, Aaron Courville, and Yoshua Bengio. ICML 2013

    This class works on 3D inputs (n_maps, n_pixels, n_pixels) only.
    """
    def __init__(self, n_in, k, **kwargs):
        """

        * `k` <int>
            The downsampling (pooling) factor (#rows, #cols)
        """
        if len(n_in) != 4:
            raise TypeError("This layer requires a batch of 3D inputs.")
        if n_in[-2] % k != 0:
            raise ValueError("To pool over k={} channels, input channels should be a multiple "
                             "of {}. The number of input channels is {}.".format(k, k, n_in[-2]))
        super(MaxPoolChannelsLayer, self).__init__(n_in, **kwargs)
        self.k = k

        self.n_out = n_in[:-3] + [n_in[-2] / k] + n_in[-2:]

    def compile(self, input):
        super(MaxPoolChannelsLayer, self).compile(input)
        raise NotImplementedError("Implement the cross-channel pooling here.")

    def __str__(self):
        return "{}: MaxPool over k={} channels layer; input shape = {}; output shape = {}".\
            format(self.name, self.k, self.n_in, self.n_out)

    def get_params(self):
        """Parameters required by __init__"""
        param_dict = super(MaxPoolChannelsLayer, self).get_params()
        param_dict["k"] = self.k
        return param_dict


class MaxPool2DLayer(PassthroughLayer):
    """Pool Layer of a convolutional network.
    This uses "max pooling". For each `pool_shape` area in the input
    image, we keep only the maximum value.

    This class works on 2D inputs."""
    def __init__(self, n_in, pool_shape, stride=None, **kwargs):
        """

        **Optional Parameters**

        * `pool_shape` <tuple or list of length 2>
            The downsampling (pooling) factor (#rows, #cols)

        **Optional Parameters**

        * `stride` <tuple or list of length 2|None>
            Take steps of this size across the input image. If None, the stride
             equals the pool_shape.
        """
        if stride is None:
            stride = pool_shape
        if isinstance(stride, int):
            stride = (stride, stride)

        if CUDA_CONVNET:
            if pool_shape[0] != pool_shape[1]:
                raise ValueError("For optimized max-pooling, we can only pool over square regions.")
            if stride[0] != stride[1]:
                raise ValueError("For optimized max-pooling, strides "
                                 "must be equal in each dimension.")
            if n_in[0] % 4 != 0 and n_in[0] > 4:
                raise ValueError("Cuda-convnet requires a number of input "
                                 "channels divisible by or less than 4.")

        super(MaxPool2DLayer, self).__init__(n_in, **kwargs)
        self.pool_shape = pool_shape
        self.stride = stride
        self.n_out = n_in[:-2] + (int(np.ceil(n_in[-2] / pool_shape[0])),
                                  int(np.ceil(n_in[-1] / pool_shape[1])))

    def compile(self, input):
        super(MaxPool2DLayer, self).compile(input)
        # Downsample each feature map individually using maxpooling.
        if CUDA_CONVNET:
            pool_op = pool.MaxPool(ds=self.pool_shape[0], stride=self.stride[0])
            self.output = pool_op(gpu_contiguous(self.input))
        else:
            self.output = downsample.max_pool_2d(input=self.input, ds=self.pool_shape,
                                                 st=self.stride)

    def __str__(self):
        return "{}: MaxPooling {} layer with stride {}; input shape = {}; output shape = {}".\
            format(self.name, self.pool_shape, self.stride, self.n_in, self.n_out)

    def get_params(self):
        """Parameters required by __init__"""
        param_dict = super(MaxPool2DLayer, self).get_params()
        param_dict["pool_shape"] = self.pool_shape
        param_dict["stride"] = self.stride
        return param_dict


class BC01ToC01BLayer(PassthroughLayer):
    """
    Shuffle an input's ordering from BC01 (used in Theano) to C01B (used by cuda-convnet).
    The "B" is the minibatch size, the "C" is the number of channels, and "01" is the image size.
    This layer converts an input into something which can be used by cuda-convnet optimized
    convolutions.
    """
    def __init__(self, n_in, **kwargs):
        """
        **Parameters**

        * `input` <theano.tensor.dmatrix>
            A symbolic tensor of shape (n_examples, n_in)
        * `n_in` <int>
            Dimensionality of input
        """
        super(BC01ToC01BLayer, self).__init__(n_in, **kwargs)

    def compile(self, input):
        super(BC01ToC01BLayer, self).compile(input)

        self.output = self.input.dimshuffle(1, 2, 3, 0)  # bc01 to c01b

    def __str__(self):
        return "Theano-to-cuda-convnet dimshuffle."


class C01BToBC01Layer(PassthroughLayer):
    """
    Shuffle an input's ordering from C01B (used by cuda-convnet) to BC01 (used in Theano).
    The "B" is the minibatch size, the "C" is the number of channels, and "01" is the image size.
    This layer converts the output of cuda-convnet optimized convolutions to something expected
    by Theano code.
    """
    def __init__(self, n_in, **kwargs):
        """
        **Parameters**

        * `input` <theano.tensor.dmatrix>
            A symbolic tensor of shape (n_examples, n_in)
        * `n_in` <int>
            Dimensionality of input
        """
        super(C01BToBC01Layer, self).__init__(n_in, **kwargs)

    def compile(self, input):
        super(C01BToBC01Layer, self).compile(input)

        self.output = self.input.dimshuffle(3, 0, 1, 2)  # c01b to bc01

    def __str__(self):
        return "Cuda-convnet-to-Theano dimshuffle."
