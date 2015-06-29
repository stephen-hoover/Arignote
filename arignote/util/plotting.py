"""
Utilities for plotting neural networks.
"""
__author__ = 'shoover'

from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns


def show_cost(metadata, plot_train=True, axes=None):
    train_iter, train_cost = np.asarray(metadata["all_train_cost"]).T
    valid_iter, valid_cost = np.asarray(metadata["all_valid_loss"]).T

    # Create the canvas if we weren't given one.
    if axes is None:
        _, axes = plt.subplots()

    if plot_train:
        axes.plot(train_iter, train_cost, label="Training cost (NLL + reg)")
    axes.plot(valid_iter, valid_cost, label="NLL on validation set", linewidth=4)
    axes.set_xlabel("Minibatch number")
    axes.set_ylabel("Loss")

    legend = plt.legend()

    return axes