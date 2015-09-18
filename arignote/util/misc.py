"""
Utility functions which don't quite fit anywhere else.
"""
from __future__ import division

__author__ = 'shoover'

import collections


import numpy as np
import six
import theano
from theano import tensor as T

from ..util import netlog
log = netlog.setup_logging("nnets_misc", level="INFO")


def get_tensor_type_from_data(arr, name=None, dtype=None):
    """Returns the Theano tensor variable type corresponding to the input data.
    E.g., a 2-dimensional float array input results in a T.matrix output.
    Override the array's native data type with a `dtype` input."""
    arr = np.asarray(arr)
    if dtype is None:
        dtype = str(arr.dtype)
        if dtype.startswith("float"):
            dtype = theano.config.floatX
    return T.TensorType(dtype, broadcastable=arr.ndim * (False,))(name)


def stratified_holdout(labels, frac, extra_frac=False, rng=None):
    """Given input labels (1D), returns indices of a holdout set making up a random fraction
    `frac` of each class in the input array. Given an `extra_frac` input, will also return
    a second, non-overlapping array.
    """
    # Initialize a random number generator, if not provided.
    if not isinstance(rng, np.random.RandomState):
        log.debug("Making a new RNG for the stratified shuffle with seed {}.".format(rng))
        rng = np.random.RandomState(rng)

    labels = np.asarray(labels)  # Needs to be an array.

    label_counter = collections.Counter(labels)

    # Select a random `frac` of each of the labels.
    holdout = []
    for name, num in label_counter.items():
        holdout.extend(rng.choice(np.where(labels == name)[0], frac * num, replace=False).tolist())
    holdout = np.asarray(holdout)

    # Select a disjoint subset of fractional size `extra_frac`, if provided.
    if extra_frac is not False:
        chosen = np.zeros(len(labels), dtype=bool)
        chosen[holdout] = True
        remaining_labels = labels[~chosen]
        i_extra_in_subset = stratified_holdout(remaining_labels, extra_frac / (1 - frac), rng=rng)
        extra_holdout = np.arange(len(labels))[~chosen][i_extra_in_subset]

        return holdout, extra_holdout
    else:
        return holdout


def stratified_shuffle(labels, batch_size, balance_classes=False, rng=None):
    """Returns indices which will shuffle the input `labels` (1D integer array) into
    batches of size `batch_size`, in which the batches have as even a distribution of labels
    as possible.
    """
    # Initialize a random number generator, if not provided.
    if not isinstance(rng, np.random.RandomState):
        log.debug("Making a new RNG for the stratified shuffle with seed {}.".format(rng))
        rng = np.random.RandomState(rng)

    labels = np.asarray(labels)  # Needs to be an array.

    label_counter = collections.Counter(labels)
    max_examples = max(label_counter.values())
    all_rows = {lab: np.where(labels == lab)[0] for lab in label_counter}
    for label in all_rows:
        rng.shuffle(all_rows[label])
        all_rows[label] = all_rows[label].tolist()
    if balance_classes:
        # Upsample each class so that all classes have the same number of examples.
        for y, rows in all_rows.items():
            n_needed = max_examples - label_counter[y]
            rows.extend(rng.choice(rows, n_needed, replace=True))
        label_counter = {y: len(rows) for y, rows in all_rows.items()}

    # Instantiate the bins
    nbins = int(np.floor(sum(label_counter.values())/float(batch_size)))
    bins = {}
    for k in range(nbins):
        bins[k] = []
    final_bin = nbins

    # Evenly distribute each class across the bins
    non_full_bins = [k for k in bins]
    for y, rows in all_rows.items():
        unused_bins = []
        for row in rows:
            if len(unused_bins) == 0:
                unused_bins = [x for x in non_full_bins]
            k = int(rng.choice(unused_bins))
            bins[k].append(row)
            unused_bins.remove(k)
            if len(bins[k]) == batch_size:
                non_full_bins.remove(k)

                # Add remaining rows to final bucket
                if len(non_full_bins) == 0:
                    bins[final_bin] = []
                    non_full_bins.append(final_bin)

    # Buffer the final bucket to have size batch_size
    if len(bins[final_bin]) < batch_size:
        samp = batch_size - len(bins[final_bin])
        buff = rng.choice(np.arange(len(labels)), samp).tolist()
        bins[final_bin].extend(buff)

    # Get the indices (order) of the shuffled data frame
    indices = []
    for val in bins.values():
        indices.extend(rng.permutation(val).tolist())

    return indices


def as_list(x):
    """If an object is a string or non-iterable, returns it as a one-element list.
    Otherwise returns the object unchanged.
    """
    if isinstance(x, six.string_types) or not isinstance(x, collections.Iterable):
        x = [x]
    return x


def is_listlike(x):
    if isinstance(x, collections.Iterable) and not isinstance(x, six.string_types):
        return True
    else:
        return False


def is_floatlike(x):
    try:
        float(x)
        return True
    except TypeError:
        return False


def flatten(x):
    """Flatten a list of arbitrary depth. Returns a list with no sub-lists or sub-tuples.
    If the input is not a list or a tuple, it will be returned as a one-element list.
    """
    if not isinstance(x, (list, tuple)):
        return [x]
    else:
        if len(x) == 0:
            return []
        else:
            return flatten(x[0]) + flatten(x[1:])
