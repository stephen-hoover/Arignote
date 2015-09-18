# Arignote

From Wikipedia: Arignote or Arignota (Greek: Ἀριγνώτη, Arignṓtē) was a Pythagorean philosopher who flourished around the year 500 bc. She was known as a student of Pythagoras and Theano and, according to some traditions, their daughter as well.

# Introduction

Arignote is a library for building and training neural networks, built on the Theano library.
It's built to be flexible and extendable. Arignote trains neural networks using minibatch SGD,
so you're not required to have all data in memory at once. The checkpointing feature allows
training to stop and resume as if it had never been interrupted (useful for working with EC2
spot instances).

Warning -- this library currently has no tests (so is undoubtedly broken in some way).
The code is also rough in many places. At this time, Arignote only supports feed-forward
networks; recurrent neural networks may or may not be added in the future.

# Installation

Clone the Arignote repository, then run
```
python setup.py install
```
or
```
python setup.py develop
```

## Requirements

- Tested and known to work with Python 3.4 and 2.7.10
- theano
- numpy
- six

Optional requirements
- pandas (needed to read HDF5 files)


# Instructions

## Network initialization

Define your network with a list of layers. Each element of the list must be a list or
tuple with two elements: The type of the layer (a string), and a dictionary of
parameters for that layer.

This is an example definition of a fully-connected layer. It has an (optional) name,
a number of neurons (required), an activation function (defaults to "prelu"), and
a ridge regularization constant (defaults to 0).
```
["FCLayer", {"name": "fc1", "n_units": 100, "activation": "relu", "l2": 0.001}]
```

All layers take the following parameters:
- name (optional): A layer name for debugging and display.

All layers with weights take the following parameters:
- l1 (0): L1 weight decay
- l2 (0): L2 weight decay
- maxnorm (None): Maximum norm of a row in the weight array. Rows which would become larger during training will be scaled to fit.

Available trainable layer types and their parameters are:
- "ClassificationOutputLayer" (A multi-output logistic regression layer)
    - n_classes : int, number of output classes (softmax output)
- "FCLayer" (Fully-connected layer)
    - n_units
    - activation ("prelu"): str, Name of the activation function
- "FCMaxoutLayer" (Fully-connected layer using the "maxout" activation function)
    - n_units
    - maxout_k (5) : int, Number of linear activations in the maxout
- "ConvLayer" (convolutional layer)
    - n_output_maps : int, Number of output kernels
    - filter_shape : length-2 list of ints, size of convolutional filter in pixels
    - activation ("prelu"): str
    - stride (1, 1): length-2 list of ints, non-default values may not be working
- "ConvMaxoutLayer" (convolutional layer with "maxout" activation function)
    - n_output_maps : int, Number of output kernels
    - filter_shape : length-2 list of ints, size of convolutional filter in pixels
    - maxout_k (5): int, Number of linear activations in the maxout
    - stride (1, 1): length-2 list of ints, non-default values may not be working
- "MLPConvLayer" ("mlpconv" layer from arXiv:1312.4400v3)
    - n_output_maps : int, Number of output kernels
    - filter_shape : length-2 list of ints, size of convolutional filter in pixels
    - n_units : list of integers, number of neurons in each layer of the mlpconv activation
    - activation ("prelu"): str
    - stride (1, 1): length-2 list of ints, non-default values may not be working

Non-trainable layer types:
- "DropoutLayer"
    - dropout_p : float between 0 and 1. Dropout this fraction of the previous layer's outputs.
- "MaxPool2DLayer"
    - pool_shape : length-2 list of ints
    - stride (None) : Defaults to `pool_shape`

Utility layers:
- "InputImageLayer" (Required if input is more than 1D)
    - n_images : int, number of channels in the input
    - n_pixels : length-2 list of ints, number of pixels in each image
- "BC01ToC01BLayer" (Place before a block of convolutions. Required for CUDA-Convnet optimizations)
- "C01BToBC01Layer" (Place after a block of convolutions. Required for CUDA-Convnet optimizations)


## Data input

Arignote can take dense arrays of floating-point data, or HDF5 files containing dense arrays
of floating-point data.

You can create HDF5 files using pandas, for example:
```
import pandas as pd

store = pd.HDFStore(filename)
for i_row in range(n_images):
        image, label = read_image_from_somewhere(i_row)

        # "label" should be a list or array.
        store.append("labels", label)
        store.append("images", pd.Panel4D({i_row: image}),
                     axes=["labels", "major_axis", "minor_axis"], complib="zlib", complevel=9)

store.close())
```

Then create an Arignote data reader with e.g.
```
from arignote.data import readers

features = readers.HDFReader(filename, "images", asarray=True)
labels = readers.HDFReader(filename, "labels", asarray=False)
data = readers.DataWithHoldoutParitions(features, labels, batch_size=batch_size,
                                        valid_frac=valid_frac, test_frac=test_frac)
```

You can create your own data input object to feed in data from arbitrary data sources.
Just subclass the `readers.Reader` abstract class and define the method `iter_epoch` which
takes arguments `batch_size, start=0, stop=None, start_on_batch=True, allow_partial=False`.
See the `readers` module for more.

Note that the `reader.Data` and `reader.DataWithHoldoutPartitions` also accept functions
which can alter each minibatch on readout, for example for data augmentation.
This function will run inside a thread, so that it can operate during GPU training
(theano releases the GIL, and your CPUs aren't doing anything else while the GPU is working).

## Training

Now that you've defined your network and gotten your data sources, it's time to
train the network. Instantiate a new network with
```
from arignote.nnets import nets

classifier = nets.NNClassifier(layer_definition_list, name="Sample network", rng=42)
```

and fit it with `classifier.fit`. See the docstrings in the `nets.NNClassifier` class
for details of the usage and all of the arguments available.

### The `fit` function

Perform supervised training on the input data.

When restoring a pickled `NNClassifier` object to resume training,
data, augmentation functions, and checkpoint locations must be
re-entered, but other parameters will be taken from the previously
stored training state. (The `n_epochs` may be re-supplied to alter
the number of epochs used, but will default to the previously
supplied `n_epochs`.)

Training may be stopped early by pressing ctrl-C.

Training data may be provided in either of the following formats:
- An array of (n_examples, n_features) in the first positional
    argument (keyed by `X`), and an array of (n_examples, n_labels)
    in the second positional argument (keyed by `y`)
- An object of type `readers.DataWithHoldoutParitions` or `readers.Data`
    presented in the first positional argument

Validation data may be optionally supplied with the `valid` key
in one of the following formats (only if the training data were not
given as a `readers.DataWithHoldoutParitions` object):
- A tuple of (X, y), where `X` is an array of
    (n_validation_examples, n_features) and `y` is an array of
    (n_validation_examples, n_labels)
- A `readers.Data` object
- A float in the range [0, 1), in which case validation data will
    be held out from the supplied training data (only if training
    data were given as an array)

Test data may be optionally supplied with the `test` key, using the same
formats as for validation data.

    Parameters
    ----------
    X, y, valid, test
        See above for discussion of allowed input formats.
    n_epochs : int
        Train for this many epochs. (An "epoch" is one complete pass through
        the training data.) Must be supplied unless resuming training.
    batch_size : int
        Number of examples in a minibatch. Must be provided if was
        not given during object construction.
    augmentation : function, optional
        Apply this function to each minibatch of training data.
    checkpoint : str, optional
        Filename for storing network during training. If supplied,
        Arignote will store the network after every epoch, as well
        as storing the network with the best validation loss and
        the final network. When using a checkpoint, the trainer
        will restore the network with best validation loss at the
        end of training.
    sgd_type : {"adadelta", "nag", "adagrad", "rmsprop", "sgd"}
        Choice for stochastic gradient descent algorithm to use in training
    lr_rule, momentum_rule : dict of sgd_updates.Rule params, optional
        Use these dictionaries of parameters to create Rule objects
        which describe how to alter the learning rate and momentum
        during training.
    train_loss, valid_loss : {"nll", "error"}
        Loss function for training and validation. With a custom
        output layer, may also be the name of a function which returns
        a theano symbolic variable giving the cost.
        ("nll" = "negative log likelihood")
    test_loss : str or list
        May be any of the loss functions usable for training, or
        a list of such functions.

    Other Parameters
    ----------------
    sgd_max_grad_norm : float, optional
        If provided, scale gradients during training so that the norm
        of all gradients is no more than this value.
    validation_frequency : int, optional
        Check the validation loss after training on this many examples.
        Defaults to validating once per epoch.
    validate_on_train : bool, optional
        If set, calculate validation loss (using the deterministic
        network) on the training set as well.
    checkpoint_all : str, optional
        Keep the state of the network at every training step.
        Warning: may use lots of hard drive space.
    extra_metadata : dict, optional
        Store these keys with the pickled object.


### SGD Parameters

Training uses stochastic gradient descent. You can control the parameters of the SGD
by defining rules for learning rate and (if applicable to the SGD algorithm) momentum.
These are defined in the docstring of the `arignote.nnets.sgd_update.Rule` class, and are:

    rule : {"const", "constant", "fixed", "stalled"}
        Type of rule to apply
    initial_value : float
        Where to start the parameter
    final_value : float, optional
        Stop changing the parameter when it reaches this value.
    decrease_by : float, optional
        When changing this parameter, subtract this value from it.
    multiply_by : float, optional
        When changing this parameter, multiply by this value.
        This is done after subtracting `decrease_by`.
    interval : int, optional
        Update the parameter after this many validations.
        Typically, we validate once per epoch.
        If a `schedule` is also provided, use the `interval` only
        once the schedule is exhausted.
    schedule : list of ints, optional
        Alter the parameter at these set epochs.


## Example of use

The following example instantiates and trains a simple multi-layer perceptron on the MNIST data.

```
from arignote.nnets import nets
from arignote.data import files
from arignote import sample_data

layers = [["InputLayer", {"name": "input"}],  # May specify explicitly or leave this off.
          ["FCLayer", {"name": "fc1", "n_units": 400, "activation": "prelu_shelf", "l2": 0.001}],
          ["DropoutLayer", {"name": "DO-fc1", "dropout_p": 0.5}],
          ["FCLayer", {"name": "fc2", "n_units": 400, "activation": "prelu_shelf", "l2": 0.001}],
          ["DropoutLayer", {"name": "DO-fc2", "dropout_p": 0.5}],
          ["ClassificationOutputLayer", {"name": "output", "n_classes": 10}]]

classifier = nets.NNClassifier(layers, name="MNIST MLP", rng=42)

# Specify how the learning rate changes during training.
lr_rule = {"rule": "stalled", "initial_value": 0.1, "multiply_by": 0.25, "interval": 5}

# Specify how the momentum changes during training.
momentum_rule = {"rule": "stalled", "initial_value": 0.7, "decrease_by": -0.1,
                 "final_value": 0.95, "interval": 5}

mnist_data = files.read_pickle(sample_data.mnist)
classifier.fit(mnist_data[0], n_epochs=50, valid=mnist_data[1], test=mnist_data[2],
               augmentation=None, checkpoint=checkpoint, sgd_type="nag",
               lr_rule=lr_rule, momentum_rule=momentum_rule, batch_size=128,
               train_loss="nll", valid_loss="nll", test_loss=["nll", "error"])
```

## Credits

Arignote began with code from the tutorials published by the University of Montreal's LISA lab,
http://deeplearning.net/tutorial/. It's taken inspiration and sometimes snippets of code from
existing neural network libraries such as Lasagne (https://github.com/Lasagne/Lasagne).
I've tried to add credits to code which was copied directly or nearly directly from elsewhere.
Functions directly inspired by a particular research paper contain a reference to that paper.

Arignote contains a copy of the MNIST dataset (http://yann.lecun.com/exdb/mnist/), converted
to a Python pickle file by the LISA lab: http://deeplearning.net/tutorial/gettingstarted.html#mnist-dataset .
