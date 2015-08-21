"""
Train a network on the MNIST data.
"""
__author__ = 'shoover'


import numpy as np

from arignote.data import files
from arignote.data import readers
from arignote.image import augment
from arignote.nnets import nets
from arignote import sample_data


def load_and_train(fname, checkpoint=None):
    """Test reloading a model in progress and continuing."""
    mnist_data = files.read_pickle(sample_data.mnist)

    c = files.checkpoint_read(fname)
    c.fit(mnist_data[0], valid=mnist_data[1], test=mnist_data[2], checkpoint=checkpoint)


def fit_mnist_small_mlp(n_epochs, checkpoint=None):
    """Train a one-layer MLP using MNIST data. Use for testing.
    """
    lr_rule = {"rule": "stalled", "initial_value": 0.1, "multiply_by": 0.25, "interval": 5}
    momentum_rule = {"rule": "stalled", "initial_value": 0.7, "decrease_by": -0.1,
                     "final_value": 0.95, "interval": 5}

    # Demonstrate automatic partitioning of data into training, validation, and test sets.
    mnist_data = files.read_pickle(sample_data.mnist)
    mnist_features = np.concatenate([mnist_data[0][0], mnist_data[1][0], mnist_data[2][0]])
    mnist_labels = np.concatenate([mnist_data[0][1], mnist_data[1][1], mnist_data[2][1]])

    layers = [["FCLayer", {"name": "fc2", "n_units": 100, "activation": "prelu_shelf", "l2": 0.001}],
              ["DropoutLayer", {"name": "DO-fc2", "dropout_p": 0.5}],
              ["ClassificationOutputLayer", {"name": "output", "n_classes": 10}]]

    classifier = nets.NNClassifier(layers, name="MNIST MLP", random_state=42)
    classifier.fit(mnist_features, mnist_labels, valid=1/7, test=1/7, n_epochs=n_epochs,
                   augmentation=None, checkpoint=checkpoint, sgd_type="nag",
                   lr_rule=lr_rule, momentum_rule=momentum_rule, batch_size=100,
                   train_loss="nll", valid_loss="nll", test_loss=["nll", "error"])

    return classifier


def fit_mnist_mlp(checkpoint=None):
    """Train a three-layer MLP using MNIST data.

    128-128-10, prelu_shelf: 2% error
    400-400-10, prelu_shelf: 1.55% error
    """
    layers = [["InputLayer", {"name": "input"}],  # May specify explicitly or leave this off.
              ["FCLayer", {"name": "fc1", "n_units": 400, "activation": "prelu_shelf", "l2": 0.001}],
              ["DropoutLayer", {"name": "DO-fc1", "dropout_p": 0.5}],
              ["FCLayer", {"name": "fc2", "n_units": 400, "activation": "prelu_shelf", "l2": 0.001}],
              ["DropoutLayer", {"name": "DO-fc2", "dropout_p": 0.5}],
              ["ClassificationOutputLayer", {"name": "output", "n_classes": 10}]]

    classifier = nets.NNClassifier(layers, name="MNIST MLP", random_state=42)

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

    return classifier


def fit_mnist_conv(checkpoint=None):
    """Train a convolutional neural network using MNIST data.

    (3x3)x32 - (maxpool 2) - (2x2)x32 - (2x2)x32 - (maxpool2) - 400 - 400 - 10 :
        Trains to 0.71% error after 148 epochs.
    """
    batch_size = 128  # Let the Readers know how many images to supply with each epoch.

    # Set up the data inputs. Demonstrate explicit use of the "Data" object, and the
    # ability to alter data as it's read out. In this case, we want to reshape the
    # training images to be square so that they're easier to shift as part of data augmentation.
    def make_square(img_batch):
        return img_batch.reshape((-1, 28, 28))
    mnist_data = files.read_pickle(sample_data.mnist)
    train = readers.Data(*mnist_data[0], batch_size=batch_size, alter_features=make_square)
    valid = readers.Data(*mnist_data[1], batch_size=batch_size, alter_features=make_square,
                         allow_partial_batch=True)
    test = readers.Data(*mnist_data[2], batch_size=batch_size, alter_features=make_square,
                        allow_partial_batch=True)

    # Define the network architecture.
    layers = [  # Now an input layer is needed to translate the flat array to a square image.
        ["InputImageLayer", {"name": "input", "n_images": 1, "n_pixels": (28, 28)}],
        ["BC01ToC01BLayer"],  # Needed for optional running with CUDA-convnet.
        ["ConvLayer", {"name": "conv1", "n_output_maps": 32, "filter_shape": (3, 3),
                       "activation": "prelu"}],
        ["MaxPool2DLayer", {"name": "pool1", "pool_shape": (2, 2)}],
        ["ConvLayer", {"name": "conv2", "n_output_maps": 32, "filter_shape": (2, 2),
                       "activation": "prelu"}],
        ["ConvLayer", {"name": "conv3", "n_output_maps": 32, "filter_shape": (2, 2),
                       "activation": "prelu"}],
        ["MaxPool2DLayer", {"name": "pool3", "pool_shape": (2, 2)}],
        ["C01BToBC01Layer"],  # Switch back to Theano ordering
        ["FCLayer", {"name": "fc1", "n_units": 400, "activation": "prelu_shelf", "l2": 0.01}],
        ["DropoutLayer", {"name": "DO-fc1", "dropout_p": 0.5}],
        ["FCLayer", {"name": "fc2", "n_units": 400, "activation": "prelu_shelf", "l2": 0.01}],
        ["DropoutLayer", {"name": "DO-fc2", "dropout_p": 0.5}],
        ["ClassificationOutputLayer", {"name": "output", "n_classes": 10}]
        ]

    # Instantiate the network and start training.
    classifier = nets.NNClassifier(layers, n_in=train.features.shape, name="MNIST ConvNet",
                                   batch_size=batch_size, random_state=42)

    lr_rule = {"rule": "stalled", "initial_value": 0.05, "multiply_by": 0.5,
               "interval": 8, "final_value": 1e-3}
    momentum_rule = {"rule": "stalled", "initial_value": 0.5, "decrease_by": -0.1,
                     "final_value": 0.95, "interval": 8}
    classifier.fit(train, n_epochs=200, valid=valid, test=test,
                   augmentation=augment.create_shift_augment(3, 3),
                   checkpoint=checkpoint, sgd_type="nag",
                   lr_rule=lr_rule, momentum_rule=momentum_rule, batch_size=batch_size,
                   train_loss="nll", valid_loss="nll", test_loss=["nll", "error"])

    return classifier
