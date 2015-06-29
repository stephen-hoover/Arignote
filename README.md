# Arignote

From Wikipedia: Arignote or Arignota (Greek: Ἀριγνώτη, Arignṓtē) was a Pythagorean philosopher who flourished around the year 500 bc. She was known as a student of Pythagoras and Theano and, according to some traditions, their daughter as well.

## Introduction

Arignote is a library for building and training neural networks, built on the Theano library.
It's built to be flexible and extendable. Arignote trains neural networks using minibatch SGD,
so you're not required to have all data in memory at once. The checkpointing feature allows
training to stop and resume as if it had never been interrupted (useful for working with EC2
spot instances).

Arignote is not a finished product -- calling it "alpha" would be too generous. It's useful,
but currently has no tests (so is undoubtedly broken in some way). The code is also rough in
many (most?) places. It currently contains only feed-forward networks; recurrent neural
networks may or may not be added in the future.

## Installation

Clone the Arignote repository, then run
```
python setup.py install
```
or
```
python setup.py develop
```

## Instructions

### Network initialization

### Data input

### Training


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
