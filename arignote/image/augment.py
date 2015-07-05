"""
This module controls image augmentation. Functions here will transform images into different
images which still represent the desired class.
"""
from __future__ import division

__author__ = 'shoover'


import numpy as np
from skimage.io import imread


from . import process


from ..util import netlog
log = netlog.setup_logging("image_augment", level="INFO")


def no_augmentation(x, y=None, epoch=0, rng=None, **kwargs):
    """If the trainer isn't supplied a function for data augmentation,
    use this blank function instead."""
    if y is None:
        return x
    else:
        return x, y


flip_transforms = [np.fliplr, lambda x: x]
rot_transforms = [lambda x: np.rot90(x, 1),
                  lambda x: np.rot90(x, 2),
                  lambda x: np.rot90(x, 3),
                  lambda x: x]
def random_transformation(X, rng):
    size = np.sqrt(len(X))
    flipped_img = rng.choice(flip_transforms)(X.reshape([size, size]))
    rot_img = rng.choice(rot_transforms)(flipped_img)
    return rot_img.flatten()


def augmentation_pipeline(*augment_funcs):
    """Takes an arbitrary number of functions with input (X, y, epoch, rng) and output (X, y),
    and returns a single function with the same signature which applies the input
    functions in order."""
    def augmentation_func(train_x, train_y, epoch, rng):
        for func in augment_funcs:
            train_x, train_y = func(train_x, train_y, epoch=epoch, rng=rng)
        return train_x, train_y

    return augmentation_func


def alter_training_minibatch(train_x, train_y, epoch, rng):
    """Dynamic data augmentation. Returns the training images randomly flipped and rotated,
    along with unaltered labels.

    **Parameters**

    * `train_x` <2D np.array>
        A minibatch with shape (batch_size, n_inputs)
    * `train_y` <1D np.array>
        Labels for each example in `train_x`. Has size (batch_size,).
    * `iteration` <int>
        Epoch or iteration number for this training set, in case we're altering deterministically.

    **Returns**

    A 2-tuple of arrays.
    """
    train_x = [random_transformation(x, rng) for x in train_x]
    return train_x, train_y

def create_shift_augment(max_shift_x, max_shift_y=None):
    """A factory function which creates and returns a data augmentation function. The returned
    data augmentation function will randomly shift a minibatch of input images by a
    number of pixels in the X and Y directions, with the number of pixels for the shift drawn
    from a uniform distribution."""
    if max_shift_y is None:
        max_shift_y = max_shift_x

    def shift_augment(train_x, train_y, epoch, rng):
        shift_x = rng.random_integers(-max_shift_x, max_shift_x)
        shift_y = rng.random_integers(-max_shift_y, max_shift_y)

        # Axis 0 is the minibatch index, so axis 2 is "x" and axis 1 is "y".
        train_x = np.roll(np.roll(train_x, shift_x, axis=1), shift_y, axis=2)
        if shift_y < 0:
            train_x[:, shift_y:, :] = 0
        else:
            train_x[:, :shift_y, :] = 0

        if shift_x < 0:
            train_x[:, :, shift_x:] = 0
        else:
            train_x[:, :, :shift_x] = 0

        return train_x, train_y

    return shift_augment
