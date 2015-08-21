"""
Take a model and data, and use minibatch stochastic gradient descent to fit the model to the data.
"""
from __future__ import division

__author__ = 'shoover'


import abc
from datetime import datetime, timedelta
import os
import time
import six

import numpy as np
import theano
import theano.tensor as T

from ..data import files
from ..image import augment
from ..nnets import sgd_updates
from ..util import misc
from ..util import netlog


log = netlog.setup_logging("nnet_training", level="INFO")


def format_cost(name, val):
    """Return a string representing the cost, suitable for log output during training."""
    name, val = misc.as_list(name), misc.as_list(val)
    cost_string = []
    for n, v in zip(name, val):
        n = n.lower().replace(" ", "_")
        if n in ["nll", "negative_log_likelihood"]:
            cost_string.append("NLL is {:.5}".format(v))
        elif n in ["error"]:
            cost_string.append("error is {:.2%}".format(v))

    return "; ".join(cost_string)


class BaseTraining(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        pass


class UnsupervisedLayerwiseTraining(object):
    """ Train a neural network layer by layer using input features only, no labels.

    **Parameters**


    * `loss` <string|"squared">
        Loss function for unsupervised training. Use "cross-entropy" if the features can
        be interpreted as bits or bit probabilities. Otherwise use "squared".
    """

    def __init__(self, sgd, features=None, max_epochs=None,
                 batch_size=128, validation_frequency=10000,
                 validate_on_train=False, regularized=False,
                 loss="squared"):
        self.sgd = sgd
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.validation_frequency = validation_frequency
        self.validate_on_train = validate_on_train
        self.regularized = regularized
        self.loss = loss  # Loss function for the autoencoder loss

        # Generate symbolic variables for the input (x represents a minibatch).
        if features is None:
            features = T.matrix("features")
        self.x = features
        self.functions = []

        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []
        self.train_valid_loss = []  # Result of treating the training data as a validation set.

    def compile(self, classifier, recompile=False):
        if len(self.functions) > 0 and not recompile:
            log.debug("Functions already compiled. Not recompiling.")
            return

        log.info("Compiling layer-wise training, validation, and test functions.")
        log.info("Using \"{}\" loss functions.".format(self.loss))

        classifier.compile(self.x, recompile=recompile)  # Make sure the classifier is compiled.

        # Compile loss functions for each layer which can be trained unsupervised.
        # Use the "inference" layers -- we consider only one layer at a time, and we want all of
        # the layers beneath that layer to perform a normal forward pass when we
        # call on their outputs.
        self.functions = []
        for layer in classifier.layers_inf:
            if len(layer.params) == 0:
                continue
            if (len(layer.params) != 0 and
                  (not hasattr(layer, "get_autoencoder_loss") or not layer.may_train_unsupervised)):
                break

            this_corr_level = T.scalar("corruption_{}".format(layer.name),
                                       dtype=theano.config.floatX)
            ac_loss = layer.get_autoencoder_loss(corruption_level=this_corr_level,
                                                 regularized=self.regularized,
                                                 loss=self.loss)

            updates = self.sgd.get_updates(cost=ac_loss, params=layer.ac_params, param_rules=None)

            train_func = theano.function(inputs=[self.x, theano.Param(this_corr_level, default=layer.corruption_level)],
                                         outputs=ac_loss, updates=updates,
                                         name="ac_training_{}".format(layer.name))

            # Now make functions for validation and testing.
            ac_valid_loss = layer.get_autoencoder_loss(corruption_level=this_corr_level,
                                                       regularized=False, loss=self.loss)
            valid_func = theano.function(inputs=[self.x,
                                         theano.Param(this_corr_level, default=0)],
                                         outputs=ac_valid_loss,
                                         name="ac_validation_{}".format(layer.name))
            test_func = theano.function(inputs=[self.x,
                                        theano.Param(this_corr_level, default=0)],
                                        outputs=ac_valid_loss,
                                        name="ac_test_{}".format(layer.name))

            self.functions.append((layer.name, train_func, valid_func, test_func))

        log.info("Found {} layers for layer-wise unsupervised "
                 "training.".format(len(self.functions)))

    def fit(self, classifier, data, augmentation=None,
            checkpoint=None, checkpoint_all=False,
            extra_metadata=None):
        """
        * `train_data` <2-tuple of ndarrays>
            Assumed to be a tuple of (training_features, training_targets).
        """
        use_best = True  # Restore the state with best validation loss at the end of training.

        self.compile(classifier)

        # Allow for `augmentation` to be not given, given as a single function, or
        # given as a list of functions to be applied one after the other.
        if augmentation is None:
            augmentation = augment.no_augmentation
        augmentation = augment.augmentation_pipeline(*misc.as_list(augmentation))

        # Use validation frequency in nearest number of iterations.
        validation_frequency = self.validation_frequency // self.batch_size

        checkpoint_dir, checkpoint_fname = files.parse_checkpoint(checkpoint)
        best_checkpoint_name = ""

        def gather_checkpoint_info():
            metadata = extra_metadata if extra_metadata else {}
            metadata.update({"log": log.debug_global,
                             "last_training_loss": this_training_loss,
                             "last_validation_loss": this_validation_loss,
                             "last_test_loss": this_test_loss,
                             "best_valid_loss": best_validation_loss,
                             "epoch": epoch,
                             "examples_seen": examples_seen,
                             "seconds_elapsed": time.time() - start_time,
                             "train_loss": self.train_loss,
                             "valid_loss": self.valid_loss,
                             "test_loss": self.test_loss,
                             "train_valid_loss": self.train_valid_loss,
                             "checkpoint_stem": checkpoint_fname})
            return metadata


        log.info("Beginning unsupervised training.")
        start_time = time.time()
        epoch_global = 0  # Count epochs over all layers.
        for name, train_func, valid_func, test_func in self.functions:
            log.info("*** Begin unsupervised training of layer {}.".format(name))
            log.info("Using {}".format(self.sgd))

            # Initialize the loop parameters.
            this_test_loss, this_validation_loss = np.inf, np.inf
            this_training_loss, this_train_valid_loss = np.inf, np.inf
            best_validation_loss = np.inf
            done_training = False
            epoch, iter = 0, -1  # Note that we increment at the start of the loops.
            examples_seen = 0
            while (epoch < self.max_epochs) and (not done_training):
                epoch += 1
                epoch_global += 1
                for minibatch_index, train_batch in enumerate(data.iter_epoch("train")):
                    iter += 1  # Iteration number
                    examples_seen += train_batch.shape[0]

                    these_examples = augmentation(train_batch, epoch=epoch)
                    # Simultaneous cost computation and update!
                    this_training_loss = float(train_func(these_examples))
                    self.train_loss.append((iter, this_training_loss))

                    if (iter + 1) % validation_frequency == 0:
                        info_string = "Epoch {}, training cost {:.5}".format(epoch, this_training_loss)

                        # We might want to know what the training data look like when run through
                        # the inference net. If so, compute that now.
                        if self.validate_on_train:
                            this_train_valid_loss = np.mean([valid_func(batch) for batch in
                                                             data.iter_epoch("train")])
                            self.train_valid_loss.append((iter, this_train_valid_loss))
                            info_string += "; training inference loss {:.5}".format(this_train_valid_loss)
                        # Compute loss on the validation set:
                        if data.using_partition["valid"]:
                            this_validation_loss = np.mean([valid_func(batch) for batch in
                                                            data.iter_epoch("valid")],
                                                           axis=0)
                            self.valid_loss.append((iter, this_validation_loss))
                            info_string += "; validation loss {:.5}".format(this_validation_loss)
                        log.info(info_string)

                        # If we got the best validation score until now, improve patience and test.
                        if data.using_partition["valid"] and this_validation_loss < best_validation_loss:
                            best_validation_loss = this_validation_loss

                            # Test the model on the test set.
                            if data.using_partition["test"]:
                                this_test_loss = np.mean([test_func(batch) for batch in
                                                          data.iter_epoch("test")],
                                                         axis=0)
                                self.test_loss.append((iter, this_test_loss))
                                log.info("   Epoch {}, minibatch {}: Scoring the test set on the best "
                                         "model, loss is {:.5}.".format(epoch, minibatch_index,
                                                                        this_test_loss))

                            # This is the best model! Record it for posterity!
                            # (Alternately, if we're not using a validation set, record every new model.
                            if checkpoint is not None:
                                fname = "{}_unsup_best.pkl.gz".format(checkpoint_fname)
                                fname = os.path.join(checkpoint_dir, fname)
                                files.save_model(classifier, fname, gather_checkpoint_info())
                                log.debug("Stored the new best model to \"{}\".".format(fname))
                                best_checkpoint_name = fname

                if checkpoint:
                    # Store our progress so we can resume if interrupted or examine training performance.
                    extensions = ["last"]
                    if checkpoint_all:
                        extensions.append("epoch{}".format(epoch_global))
                    for ext in extensions:
                        fname = "{}_unsup_{}.pkl.gz".format(checkpoint_fname, ext)
                        fname = os.path.join(checkpoint_dir, fname)
                        files.save_model(classifier, fname, gather_checkpoint_info())
                        log.debug("Stored the most recent model to \"{}\".".format(fname))

                did_update = self.sgd.update_lr(epoch, this_validation_loss)
                if did_update:
                    log.info("New learning rate = {}; momentum = "
                             "{}".format(self.sgd.learning_rate.get_value(),
                                         self.sgd.momentum.get_value()))

            end_time = time.time()

            completion_string = "Optimization complete."
            if data.using_partition["valid"]:
                completion_string += " Validation loss {}.".format(this_validation_loss)
            if data.using_partition["test"]:
                completion_string += " Test loss {}.".format(this_test_loss)
            log.info(completion_string)

            if best_checkpoint_name:
                log.info("I've stored the best model at \"{}\".".format(best_checkpoint_name))
                if use_best:
                    log.info("Restoring the parameters with the best validation loss.")
                    files.restore_model(best_checkpoint_name, classifier)  # Modifies `classifier` directly.

            self.sgd.reset()  # Return learning rate and momentum to their start values.

        log.info("Finished all unsupervised training.")
        log.info("The code ran for {} epochs at {} epochs/min.".
                 format(epoch_global, 60 * epoch_global / (end_time - start_time)))

        return classifier

class SupervisedTraining(object):

    def __init__(self, sgd_type="adadelta", lr_rule=None,
                 momentum_rule=None, sgd_max_grad_norm=None,
                 max_epochs=None, validation_frequency=None, validate_on_train=False,
                 train_loss="nll", valid_loss="nll", test_loss=["error", "nll"]):
        if isinstance(sgd_type, sgd_updates.SGD):
            self.sgd = sgd_type
        else:
            self.sgd = sgd_updates.SGD(sgd_type, lr_rule, momentum_rule,
                                       max_grad_norm=sgd_max_grad_norm)
        self.max_epochs = max_epochs
        self.validation_frequency = validation_frequency
        self.validate_on_train = validate_on_train

        self.train_func = None
        self.valid_func = None
        self.test_func = None

        if not isinstance(train_loss, six.string_types):
            raise TypeError("Supply `train_loss` as a string.")
        if not isinstance(valid_loss, six.string_types):
            raise TypeError("Supply `valid_loss` as a string.")
        self.loss_type = {"train": train_loss,
                          "valid": valid_loss,
                          "test": misc.as_list(test_loss)}

        self.train_loss = []
        self.valid_loss = []
        self.test_loss = []
        self.train_valid_loss = []  # Result of treating the training data as a validation set.
        self.checkpoints_written = {"last": None, "best": None}

        # Initialize the loop parameters.
        self.last_test_loss, self.last_validation_loss = np.inf, np.inf
        self.last_training_loss, self.last_train_valid_loss = np.inf, np.inf
        self.best_validation_loss = np.inf
        self.epoch, self.iter = 0, -1  # Note that we increment at the start of the loops.
        self.examples_seen = 0
        self.time_initialized = datetime.now()
        self.time_training = timedelta()
        self.time_to_compile = timedelta()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["train_func"] = None
        state["valid_func"] = None
        state["test_func"] = None

        state["x"] = None
        state["y"] = None

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def compile(self, classifier, recompile=False, features=None, targets=None):
        if self.train_func is not None and not recompile:
            log.debug("Functions already compiled. Not recompiling.")
            return

        log.info("Compiling training, validation, and test functions.")

        start_time = datetime.now()
        if classifier.input is not None and not recompile:
            features = classifier.input
        classifier.compile(features, recompile=recompile)  # Make sure the classifier is compiled.

        # The cost we minimize during training is the negative log likelihood of
        # the model plus the regularization terms (L1 and L2); cost is expressed here symbolically.
        cost = classifier.get_loss(self.loss_type["train"], targets, regularized=True)

        updates = self.sgd.get_updates(cost=cost, params=classifier.params,
                                       param_rules=classifier.param_update_rules)

        # Compile a Theano function that returns the cost and at the
        # same time updates the parameters of the model based on the rules defined in `updates`.
        self.train_func = theano.function(inputs=[features, targets], outputs=cost,
                                          updates=updates, name="training")

        # Now compile Theano functions that compute the mistakes that are made by
        # the model on a minibatch. We may be interested in more than one kind of
        # feedback for testing.
        self.valid_func = theano.function(inputs=[features, targets], name="validation",
                outputs=classifier.get_loss(self.loss_type["valid"], targets, inference=True))
        self.test_func = theano.function(inputs=[features, targets], name="testing",
                outputs=classifier.get_loss(self.loss_type["test"], targets, inference=True))

        self.time_to_compile = datetime.now() - start_time

    def _validate(self, train, valid, test, minibatch_index):
        """Calculate loss on the current network state. If this is a new best validation
        loss, check the loss on the test set."""
        info_string = "Epoch {}, training cost {:.5}".format(self.epoch, self.last_training_loss)

        # We might want to know what the training data look like when run through
        # the inference net. If so, compute that now.
        if self.validate_on_train:
            self.last_train_valid_loss = np.mean([self.valid_func(*batch) for batch in
                                                  train.iter_epoch()], axis=0)
            self.train_valid_loss.append((self.iter, self.last_train_valid_loss))
            info_string += "; training {}".format(format_cost(self.loss_type["train"],
                                                              self.last_train_valid_loss))
        # Compute loss on the validation set:
        if valid is not None:
            self.last_validation_loss = np.mean([self.valid_func(*batch) for batch in
                                                 valid.iter_epoch()], axis=0)
            self.valid_loss.append((self.iter, self.last_validation_loss))
            info_string += "; validation {}".format(format_cost(self.loss_type["valid"],
                                                                self.last_validation_loss))
        log.info(info_string)

        # If we got the best validation score until now, improve patience and test.
        if valid is not None and self.last_validation_loss < self.best_validation_loss:
            is_best = True
            self.best_validation_loss = self.last_validation_loss

            # Test the model on the test set.
            if test is not None:
                self.last_test_loss = np.mean([self.test_func(*batch) for batch in
                                               test.iter_epoch()], axis=0)
                self.test_loss.append((self.iter, self.last_test_loss))
                log.info("   Epoch {}, minibatch {}: Scoring the test set on the best "
                         "model, {}.".format(self.epoch, minibatch_index,
                                             format_cost(self.loss_type["test"],
                                                         self.last_test_loss)))
        else:
            is_best = False

        return is_best

    def fit(self, classifier, data, n_epochs=None, valid=None, test=None, augmentation=None,
            checkpoint=None, checkpoint_all=False, extra_metadata=None):
        """
        * `data` <readers.Data>
            Assumed to be a Data object which will provide minibatches
            of (training_features, training_targets).

        **Optional Parameters**


        """
        return_best = True  # Restore the state with best validation loss at the end of training.

        # Clear the "checkpoints_written", since we haven't written anything yet.
        self.checkpoints_written = {"last": None, "best": None}

        if extra_metadata is None:
            extra_metadata = {}

        if n_epochs is None:
            n_epochs = self.max_epochs
        if n_epochs is None:
            raise ValueError("Enter a maximum number of training epochs.")

        sample_x, sample_y = data.peek()
        self.compile(classifier, features=misc.get_tensor_type_from_data(sample_x, "features"),
                     targets=misc.get_tensor_type_from_data(sample_y, "targets"))

        # Allow for `augmentation` to be not given, given as a single function, or
        # given as a list of functions to be applied one after the other.
        if augmentation is None:
            augmentation = augment.no_augmentation
        augmentation = augment.augmentation_pipeline(*misc.as_list(augmentation))

        # Use validation frequency in nearest number of iterations.
        if self.validation_frequency is None:
            validation_frequency = len(data)
        validation_frequency = validation_frequency // data.batch_size

        checkpoint_dir, checkpoint_stem = files.parse_checkpoint(checkpoint)
        if checkpoint:
            best_checkpoint_name = os.path.join(checkpoint_dir,
                                                "{}_best_model.pkl.gz".format(checkpoint_stem))
        else:
            best_checkpoint_name = None
        extra_metadata["checkpoint_stem"] = checkpoint_stem

        # Initialize the loop parameters.
        done_training = False

        log.info("Beginning training.")
        log.info("Using {}".format(self.sgd))
        epoch_start_time = datetime.now()
        while (self.epoch < n_epochs) and (not done_training):
            try:
                self.epoch += 1
                self.time_training += datetime.now() - epoch_start_time
                epoch_start_time = datetime.now()

                for minibatch_index, train_batch in enumerate(data.iter_epoch()):
                    if minibatch_index == 0:
                        log.debug("Epoch {}, batch {} has loss "
                                  "{}".format(self.epoch, minibatch_index, self.last_training_loss))
                        sample_x, sample_y = augmentation(train_batch[0][:5], train_batch[1][:5],
                                                          epoch=self.epoch,
                                                          rng=classifier.rng)
                        log.debug("Predictions vs actual: \n{}, {}\n-------".format(
                            classifier.predict_proba(sample_x), sample_y))
                    self.iter += 1  # Iteration number

                    these_examples, these_labels = augmentation(train_batch[0], train_batch[1],
                                                                epoch=self.epoch,
                                                                rng=classifier.rng)

                    # Calling `self.train_func` calculates loss on the training examples
                    # and simultaneously updates the network's weights.
                    self.last_training_loss = float(self.train_func(these_examples, these_labels))
                    self.train_loss.append((self.iter, self.last_training_loss))
                    self.examples_seen += train_batch[0].shape[0]

                    if (self.iter + 1) % validation_frequency == 0:
                        is_best = self._validate(data, valid, test, minibatch_index)
                        if is_best:
                            # This is the best model! Record it for posterity!
                            # (Alternately, if we're not using a validation set, record every new model.
                            if best_checkpoint_name is not None:
                                files.checkpoint_write(classifier, self, best_checkpoint_name, extra_metadata)
                                self.checkpoints_written["best"] = best_checkpoint_name

                if checkpoint:
                    # Store our progress so we can resume if interrupted or examine training performance.
                    extensions = ["last"]
                    if checkpoint_all:
                        extensions.append("epoch{}".format(self.epoch))
                    for ext in extensions:
                        fname = "{}_{}.pkl.gz".format(checkpoint_stem, ext)
                        fname = os.path.join(checkpoint_dir, fname)
                        files.checkpoint_write(classifier, self, fname, extra_metadata)
                        self.checkpoints_written[ext] = fname

                did_update = self.sgd.update_lr(self.epoch, self.last_validation_loss)
                if did_update:
                    log.info("New learning rate = {}; momentum = "
                             "{}".format(self.sgd.learning_rate.get_value(),
                                         self.sgd.momentum.get_value()))
            except KeyboardInterrupt:
                done_training = True
                self.epoch -= 1  # We didn't finish it, so it doesn't count.
                log.info("Ending training early after {} epochs.".format(self.epoch))
                if self.checkpoints_written["last"]:
                    log.info("The epoch {} network state is stored at "
                             "\"{}\".".format(self.epoch, self.checkpoints_written["last"]))

        self.time_training += datetime.now() - epoch_start_time

        completion_string = "Optimization complete."
        if valid is not None:
            completion_string += " Best validation {}.".format(format_cost(self.loss_type["valid"],
                                                                           self.best_validation_loss))
        if test is not None:
            completion_string += " Test {}.".format(format_cost(self.loss_type["test"],
                                                                self.last_test_loss))
        log.info(completion_string)

        if self.checkpoints_written["best"]:
            log.info("I've stored the best model at \"{}\".".format(self.checkpoints_written["best"]))
            if return_best:
                log.info("Restoring the parameters with the best validation loss.")
                classifier.set_trainable_params(self.checkpoints_written["best"])
        log.info("Total examples seen: {}".format(self.examples_seen))
        log.info("The code ran for {} epochs at {} epochs/min.".
                 format(self.epoch, 60 * self.epoch / self.time_training.total_seconds()))

        return classifier
