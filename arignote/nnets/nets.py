"""This module describes fully-functioning networks created from the pieces in `layer`.
"""
from __future__ import division, print_function

import collections
import inspect
import six

import numpy as np
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from ..data import files
from ..data import readers
from ..nnets import layers
from ..nnets import training
from ..util import misc
from ..util import netlog


log = netlog.setup_logging("nets", level="INFO")


def define_logistic_regression(n_classes, l1_reg=0, l2_reg=0):
    """
    This function is a shortcut to build the list of layer definitions (a single layer,
    in this case) for a logistic regression classifier.
    """
    # This network is only an output layer.
    layer_defs = [["ClassificationOutputLayer", {"n_classes": n_classes,
                                                 "l1": l1_reg, "l2": l2_reg}]]
    return layer_defs


def define_cnn(n_classes, input_image_shape, n_kernels, filter_scale, poolsize,
               n_hidden, dropout_p, activation="relu", l1_reg=0, l2_reg=0):
    """
    This function is a shortcut to build the list of layer definitions for
    a convolutional neural network.
    """
    # Assume input images are 2D. If the `input_image_shape` is 3 elements,
    # the first element is the number of images in the input. Otherwise, assume
    # that there's only one image in the input.
    if len(input_image_shape) == 3:
        pass
    elif len(input_image_shape) == 2:
        input_image_shape = [1] + list(input_image_shape)
    else:
        raise ValueError("The input image shape must be (n_images, n_pixels_x, n_pixels_y).")

    try:
        # Make sure that `n_hidden` is a list.
        len(n_hidden)
    except TypeError:
        n_hidden = [n_hidden]
    try:
        # Make sure that `dropout_p` is a list.
        len(dropout_p)
    except TypeError:
        dropout_p = (1 + len(n_hidden) + len(n_kernels)) * [dropout_p]
    if len(dropout_p) != len(n_kernels) + len(n_hidden) + 1:
        raise ValueError("Either specify one dropout for all layers or one dropout for "
                         "each layer (inputs + hidden layers).")
    dropout_p = dropout_p[::-1]  # Pops come from the end, so reverse this list.

    # Start by putting on the input layer.
    layer_defs = [["InputImageLayer", {"name": "input", "n_images": input_image_shape[0],
                                       "n_pixels": input_image_shape[1:]}]]
    input_do = dropout_p.pop()
    if input_do:
        layer_defs.append(["DropoutLayer", {"name": "DO-input", "dropout_p": input_do}])

    # Add convolutional layers.
    for i_conv, (kernels, filter, pool) in enumerate(zip(n_kernels, filter_scale, poolsize)):
        layer_defs.append(["ConvLayer", {"name": "conv{}".format(i_conv),
                                         "n_output_maps": kernels,
                                         "filter_shape": (filter, filter),
                                         "activation": activation}])
        if pool:
            layer_defs.append(["MaxPool2DLayer", {"name": "maxpool{}".format(i_conv),
                                                  "pool_shape": (pool, pool)}])
        layer_do = dropout_p.pop()
        if layer_do:
            layer_defs.append(["DropoutLayer", {"name": "DO-conv{}".format(i_conv),
                                                "dropout_p": layer_do}])

    # Add fully-connected layers.
    for i_hidden, hidden in enumerate(n_hidden):
        layer_defs.append(["FCLayer", {"name": "fc{}".format(i_hidden),
                                       "n_units": hidden, "activation": activation,
                                       "l1": l1_reg, "l2": l2_reg}])
        layer_do = dropout_p.pop()
        if layer_do:
            layer_defs.append(["DropoutLayer", {"name": "DO-fc{}".format(i_hidden),
                                                "dropout_p": layer_do}])

    # Put on an output layer.
    layer_defs.append(["ClassificationOutputLayer", {"n_classes": n_classes, "l1": l1_reg,
                                                     "l2": l2_reg}])

    return layer_defs


def define_mlp(n_classes, n_hidden, dropout_p, activation="relu", l1_reg=0, l2_reg=0):
    """
    This function is a shortcut to create a multi-layer perceptron classifier.
    """
    try:
        # Make sure that `n_hidden` is a list.
        len(n_hidden)
    except TypeError:
        n_hidden = [n_hidden]
    try:
        # Make sure that `dropout_p` is a list.
        len(dropout_p)
    except TypeError:
        dropout_p = (1 + len(n_hidden)) * [dropout_p]
    if len(dropout_p) != len(n_hidden) + 1:
        raise ValueError("Either specify one dropout for all layers or one dropout for "
                         "each layer (inputs + hidden layers).")
    dropout_p = dropout_p[::-1]  # Pops come from the end, so reverse this list.

    # Start by putting on dropout for the input layer (if any).
    layer_defs = []
    input_do = dropout_p.pop()
    if input_do:
        layer_defs.append(["DropoutLayer", {"name": "DO-input", "dropout_p": input_do}])

    # Add fully-connected layers.
    for i_hidden, hidden in enumerate(n_hidden):
        layer_defs.append(["FCLayer", {"name": "fc{}".format(i_hidden),
                                       "n_units": hidden, "activation": activation,
                                       "l1": l1_reg, "l2": l2_reg}])
        layer_do = dropout_p.pop()
        if layer_do:
            layer_defs.append(["DropoutLayer", {"name": "DO-fc{}".format(i_hidden),
                                                "dropout_p": layer_do}])

    # Put on an output layer.
    layer_defs.append(["ClassificationOutputLayer", {"name": "output", "n_classes": n_classes,
                                                     "l1": l1_reg, "l2": l2_reg}])

    return layer_defs


class NNClassifier(object):
    """
    This is a neural net to be used for a classification task. It's built from
    individual layers.

    **Optional Parameters**

    * `n_in` <int or tuple|None>
        The shape of the input features. If supplied here, we'll initialize the network layers
        now. Otherwise, this will be inferred from the data supplied during the `fit`
        and the network layers will be constructed at that time.

    * `stored_network` <str|None>
    """
    def __init__(self, layer_defs, n_in=None, name="Neural Network Classifier",
                 batch_size=None, stored_network=None, random_state=None, theano_rng=None):
        self.input = None
        self.trainer = None
        self.n_in = n_in
        self.layer_defs = layer_defs
        self.batch_size = batch_size
        self.stored_network = stored_network
        self.name = name
        self.layers_train, self.layers_inf = [], []
        self.l1, self.l2_sqr = 0, 0
        self.params, self.param_update_rules, self.n_params = [], [], 0

        if type(layer_defs) != list:
            raise TypeError("Please input a list of layer definitions.")

        self.set_rng(random_state, theano_rng)  # Sets instance attributes `self.random_state` and `self.theano_rng`.
        self.pickled_theano_rng = None  # Use this to restore previous parameters.

        # Define these Theano functions during the `compile` stage.
        self.p_y_given_x = None
        self.predict_proba = None

        if self.n_in is not None:
            self._build_network(self.n_in)

    def _build_network(self, n_in, batch_size=None):
        """Create and store the layers of this network, along with auxiliary information such
        as lists of the trainable parameters in the network."""
        self.n_in = misc.as_list(n_in)  # Make sure that `n_in` is a list or tuple.
        if batch_size is not None:
            self.batch_size = batch_size

        # These next attributes are creating and storing Theano shared variables.
        # The Layers contain shared variables for all the trainable parameters,
        # and the regularization parameters are sums and products of the parameters.
        self.layers_train = self.build_layers_train(self.layer_defs, self.stored_network)
        self.layers_inf = self.duplicate_layer_stack(self.layers_train)

        self.l1, self.l2_sqr = self.get_regularization(self.layers_train)

        # Collect the trainable parameters from each layer and arrange them into lists.
        self.params, self.param_update_rules, self.n_params = self.arrange_parameters(self.layers_train)
        log.info("This network has {} trainable parameters.".format(self.n_params))

    def arrange_parameters(self, layers):
        """Extract all trainable parameters and any special update rules from each Layer.
        Also calculate the total number of trainable parameters in this network.

        **Returns**

        A 3-tuple of (parameters, parameter update rules, and number of parameters).
        The first two elements are lists of equal length, and the number of parameters is
        an integer.

        **Modifies**

        None
        """
        # The parameters of the model are the parameters of the two layers it is made out of.
        params, param_update_rules = [], []
        for ly in layers:
            params += ly.params
            param_update_rules += ly.param_update_rules

        # Calculate the total number of trainable parameters in this network.
        n_params = int(np.sum([np.sum([np.prod(param.get_value().shape) for param in layer.params])
                               for layer in layers if not getattr(layer, "fix_params", False)]))

        return params, param_update_rules, n_params

    def get_regularization(self, layers):
        """Find the L1 and L2 regularization terms for this net. Combine the L1 and L2
        terms from each Layer. Use the regularization strengths stored in each Layer.
        Note that the value returned is `l2_sqr`, the sum of squares of all weights,
        times the lambda parameter for each Layer.

        **Returns**

        A 2-tuple of (l1, l2_sqr). The `l1` is the sum of absolute values of weights times
        lambda_l1 from each Layer, and `l2_sqr` is the sum of squares of weights times
        lambda_l2 from each Layer.

        **Modifies**

        None
        """
        # L1 norm; one regularization option is to require the L1 norm to be small.
        l1 = np.sum([ly.l1 for ly in layers if ly.l1 is not None])
        if not l1:
            log.debug("No L1 regularization in this model.")
            l1 = theano.shared(np.cast[theano.config.floatX](0), "zero")

        # Square of the L2 norm; one regularization option is to require the
        # square of the L2 norm to be small.
        l2_sqr = np.sum([ly.l2_sqr for ly in layers if ly.l2_sqr is not None])
        if not l2_sqr:
            log.debug("No L2 regularization in this model.")
            l2_sqr = theano.shared(np.cast[theano.config.floatX](0), "zero")

        return l1, l2_sqr

    def build_layers_train(self, layer_defs, stored_network=None):
        """Creates a stack of neural network layers from the input layer definitions.
        This network is intended for use in training.

        **Parameters**

        * `layer_defs` <list>
            A list of Layer definitions. May contain Layers, in which case they're added
            directly to the list of output Layers.

        **Optional Parameters**

        * `stored_network` <str|None>
            A filename containing a previously stored neural network. If any layer definitions
            specify that they should be initialized with weights from an existing network,
            use the weights in the `stored_network`.

        **Returns**

        A list of initialized (but not compiled) neural network Layers.

        **Modifies**

        None
        """
        if stored_network is not None:
            log.info('Reading weights from an existing network at "{}".'.format(stored_network))
            stored_network = collections.OrderedDict(files.read_pickle(stored_network)["params"])

        log.info("Building the \"{}\" network.".format(self.name))
        if isinstance(layer_defs[0], layers.InputLayer):
            layer_objs = []
        else:
            # Initialize the layers with an input layer, if we don't have one already.
            layer_objs = [layers.InputLayer(self.n_in, name="input")]

        for ly in layer_defs:
            if isinstance(ly, layers.Layer):
                # If this is already a Layer object, don't try to re-create it.
                layer_objs.append(ly)
            else:
                prev_ly = layer_objs[-1]
                if len(ly) == 1:
                    ly.append({})  # No extra layer arguments.

                layer_name = ly[0]
                if not layer_name.endswith("Layer"):
                    # All class names end with "Layer".
                    layer_name += "Layer"
                if ((layer_name.startswith("BC01ToC01B") or layer_name.startswith("C01BToBC01"))
                    and theano.config.device == "cpu"):
                    log.warning("Skipping \"{}\" reshuffling layer for "
                                "CPU training.".format(layer_name))
                    continue
                layer_kwargs = ly[1].copy()

                init_from = layer_kwargs.pop("load_params", False)
                if init_from:
                    if init_from not in stored_network:
                        raise ValueError("Couldn't find weights for layer {} in the input "
                                         "weights.".format(init_from))

                layer_type = getattr(layers, layer_name)
                if "batch_size" in inspect.getargspec(layer_type.__init__).args:
                    layer_kwargs.setdefault("batch_size", self.batch_size)
                layer_objs.append(layer_type(n_in=prev_ly.n_out, rng=self.rng,
                                             theano_rng=self.theano_rng, **layer_kwargs))

                log.info("Added layer: {}".format(str(layer_objs[-1])))
                if init_from:
                    # Copy weights from the input file into this layer.
                    for param, input_params in zip(layer_objs[-1].params,
                                                   stored_network[init_from]):
                        param.set_value(input_params[1], borrow=True)
                    log.info("Copied input parameters from layer {} to layer "
                                 "{}.".format(init_from, layer_objs[-1].name))

        return layer_objs

    def duplicate_layer_stack(self, layer_stack):
        """Creates a stack of neural network Layers identical to the input `layer_stack`, and
        with weights tied to those Layers. This is useful to, for example, create a parallel
        network to be used for inference.

        **Parameters**

        * `layer_stack` <list of Layers>
            A list of initialized Layers.

        **Returns**

        A list of initialized (but not compiled) neural network Layers.

        **Modifies**

        None
        """
        layer_objs = []
        for i_ly, ly in enumerate(layer_stack):
            layer_type = type(ly)
            layer_kwargs = ly.get_params()

            # Construct a parallel network for inference. Tie the weights to the training network.
            layer_kwargs.update(layer_stack[i_ly].get_trainable_params())
            layer_objs.append(layer_type(rng=self.rng, theano_rng=self.theano_rng, **layer_kwargs))

        return layer_objs

    def get_loss(self, name, targets=None, inference=False, regularized=None):
        """ Return a loss function.

        **Parameters**

        * `name` <str>
            Name of the loss function. One of ["nll", "error"]. May also be a list, in which
            case this function will return a list of loss functions.

        **Optional Parameters**

        * `targets` <theano symbolic variable|None>
            If None, will be initialized to a T.imatrix named "y".
        * `inference` <bool|False>
            If True, return the loss from the inference network (for e.g. model validation).
            Otherwise use the training network.
        * `regularized` <bool|None>
            Add regularization parameters to the loss? Default to True if `inference` is False
            and False if `inference` is True.

        **Returns**

        A Theano symbolic variable representing the requested loss, or a list of symbolic
        variables if `name` is list-like.

        **Raises**

        `ValueError` if the loss is not recognized.
        """
        if self.input is None:
            raise RuntimeError("Compile this network before getting a loss function.")

        if regularized is None:
            regularized = not inference

        # If we got a list as input, return a list of loss functions.
        if misc.is_listlike(name):
            return [self.get_loss(n, targets=targets, inference=inference, regularized=regularized)
                    for n in name]

        input_name = name
        name = name.lower()
        if name == "nll":
            name = "negative_log_likelihood"
        name = name.replace(" ", "_")

        if inference:
            output_layer = self.layers_inf[-1]
        else:
            output_layer = self.layers_train[-1]

        # Look for the cost function in the output layer.
        if not hasattr(output_layer, name):
            raise ValueError("Unrecognized loss function: \"{}\".".format(input_name))

        if targets is None:
            targets = T.imatrix("y")  # Labels, presented as 2D array of [int] labels
        loss = getattr(output_layer, name)(targets)
        if regularized:
            loss = loss + self.l1 + self.l2_sqr
        return loss

    def compile(self, input, recompile=False):

        if self.input is not None:
            if recompile:
                log.warning("Recompiling and resetting the existing network.")
            else:
                log.debug("This object already compiled. Not recompiling.")
                return

        self.input = input

        log.info("Compiling the \"{}\" training network.".format(self.name))
        prev_output = input
        for ly in self.layers_train:
            ly.compile(prev_output)
            ly.compile_activations(self.input)
            prev_output = ly.output

        log.info("Compiling the \"{}\" inference network.".format(self.name))
        prev_output = input
        for ly in self.layers_inf:
            ly.compile(prev_output)
            ly.compile_activations(self.input)
            prev_output = ly.output_inf

        # Allow predicting on fresh features.
        self.p_y_given_x = self.layers_inf[-1].p_y_given_x
        self.predict_proba = theano.function(inputs=[self.input], outputs=self.p_y_given_x)

        # Now that we've compiled the network, we can restore a previous
        # Theano RNG state, if any. The "pickled_theano_rng" will only be
        # non-None if this object was unpickled.
        self.set_theano_rng(self.pickled_theano_rng)
        self.pickled_theano_rng = None

    def get_init_params(self):
        return dict(n_in=self.n_in, layer_defs=self.layer_defs,
                    name=self.name, batch_size=self.batch_size,
                    stored_network=self.stored_network)

    def set_trainable_params(self, inp, layers=None):
        """Set the trainable parameters in this network from trainable parameters in an input.

        **Parameters**

        * `inp` <NNClassifier or string> : May be an existing NNClassifier, or a filename
            pointing to either a checkpoint or a pickled NNClassifier.

        **Optional Parameters**

        * `layers` <list of strings|None> : If provided, set parameters only for the layers
            with these names, using layers with corresponding names in the input.
        """
        # Get the input and check its type.
        # If the input is a string, try reading it first as a checkpoint file, and then
        # as a NNClassifier pickle.
        if isinstance(inp, six.string_types):
            try:
                inp = files.checkpoint_read(inp, get_metadata=False)
            except files.CheckpointError as err:
                inp = files.read_pickle(inp)
        if not isinstance(inp, NNClassifier):
            raise TypeError("Unable to restore weights from a \"{}\" object.".format(type(inp)))

        # Go through each layer in this object and set its weights.
        for ly in self.layers_train:
            if layers is not None and ly not in layers:
                continue

            if ly.has_trainable_params:
                ly.set_trainable_params(inp.get_layer(ly.name))
                log.debug("Set trainable parameters in layer {} from input weights.".format(ly.name))

    def get_layer(self, name, inf=False):
        """Returns the Layer object with the given name. If `inf` is True, search the
        inference Layers, otherwise search the training Layers.
        """
        layers = self.layers_inf if inf else self.layers_train
        for ly in layers:
            if ly.name == name:
                return ly
        else:
            raise ValueError("Layer \"{}\" is not present in "
                             "network \"{}\".".format(name, self.name))

    def set_rng(self, rng, theano_rng=None):
        """Set the pseudo-random number generator in this object and in all Layers of this object.
        The `rng` input may be either a numpy.random.RandomState, the result of a `get_state`
        call on such an object, or a seed.

        **Returns**

        None

        **Modifies**

        `self.rng` and `self.theano_rng` will be set with RNGs.
        Each Layer in `self.layers_train` and `self.layers_inf` will have their RNGs set
        to be the same objects as this network's new RNGs.
        """
        # Set up the random number generator, if necessary.
        if rng is None:
            log.debug("Making new NNet RNG")
            rng = np.random.RandomState()
        elif isinstance(rng, int):
            # If we got a seed as input.
            log.debug("Setting RNG seed to {}.".format(rng))
            rng = np.random.RandomState(rng)
        elif not isinstance(rng, np.random.RandomState):
            # Assume that anything else is the state of the RNG.
            log.debug("Initializing numpy RNG from previous state.")
            rng_state = rng
            rng = np.random.RandomState()
            rng.set_state(rng_state)

        if theano_rng is None:
            log.debug("Initializing new Theano RNG.")
            theano_rng = RandomStreams(rng.randint(2 ** 30))
        self.rng = rng
        self.theano_rng = theano_rng

        for ly in self.layers_train + self.layers_inf:
            ly.rng = self.rng
            ly.theano_rng = self.theano_rng

    def set_theano_rng(self, rng_state=None):
        """Set the current state of the theano_rng from a pickled state.
        Important: This can only be done after compiling the network! The Theano
        RNG needs to see where it fits in to the graph.

        http://deeplearning.net/software/theano/tutorial/examples.html#copying-random-state-between-theano-graphs
        """
        if rng_state is not None:
            for (su, input_su) in zip(self.theano_rng.state_updates, rng_state):
                su[0].set_value(input_su)

    def __getstate__(self):
        """ Preserve the object's state. Don't try to pickle the Theano objects directly;
        Theano changes quickly. Store the values of layer weights as arrays instead
        (handled in the Layers' __getstate__ functions) and
        """
        state = self.__dict__.copy()

        state["p_y_given_x"], state["predict_proba"] = None, None
        state["l1"], state["l2_sqr"] = None, None
        state["params"], state["param_update_rules"] = None, None
        state["layers_inf"] = []  # This is redundant with `layers_train`; don't save both.
        state["rng"] = self.rng.get_state()
        state["input"] = None

        # http://deeplearning.net/software/theano/tutorial/examples.html#copying-random-state-between-theano-graphs
        state["pickled_theano_rng"] = [su[0].get_value() for su in self.theano_rng.state_updates]
        state["theano_rng"] = None

        return state

    def __setstate__(self, state):
        """ Allow unpickling from stored weights.
        """
        self.__dict__.update(state)

        # Reconstruct this object's RNG.
        # The theano_rng won't be completely reconstructed until we recompile the network.
        self.set_rng(self.rng, self.theano_rng)

        # Rebuild everything we had to take apart before saving. Note that we'll
        # still need to call `compile` to make the network fully operational again.
        self.layers_inf = self.duplicate_layer_stack(self.layers_train)
        self.l1, self.l2_sqr = self.get_regularization(self.layers_train)

        # Collect the trainable parameters from each layer and arrange them into lists.
        self.params, self.param_update_rules, self.n_params = self.arrange_parameters(self.layers_train)

    def fit(self, X, y=None, n_epochs=None, valid=None, test=None, augmentation=None,
            checkpoint=None, checkpoint_all=False, extra_metadata=None,
            sgd_type="adadelta", lr_rule=None,
            momentum_rule=None, sgd_max_grad_norm=None,
            batch_size=None, validation_frequency=None, validate_on_train=False,
            train_loss="nll", valid_loss="nll", test_loss=["error", "nll"]):
        """Perform supervised training on the input data."""
        if batch_size is None:
            batch_size = self.batch_size

        # If the inputs are not `Data` objects, we need to wrap them before
        # the Trainer can make use of them.
        train_data = X if y is None else (X, y)
        train, valid, test = readers.to_data_partitions(train_data, valid, test, batch_size=batch_size)

        # If we didn't previously know how many features to expect in the input, we can now
        # build the layers of this neural network.
        if self.n_in is None:
            self._build_network(train.features.shape, batch_size=batch_size)

        if self.trainer is not None:
            trainer = self.trainer
        else:
            trainer = training.SupervisedTraining(sgd_type=sgd_type,
                                                  lr_rule=lr_rule,
                                                  momentum_rule=momentum_rule,
                                                  sgd_max_grad_norm=sgd_max_grad_norm,
                                                  max_epochs=n_epochs,
                                                  validation_frequency=validation_frequency,
                                                  validate_on_train=validate_on_train,
                                                  train_loss=train_loss,
                                                  valid_loss=valid_loss,
                                                  test_loss=test_loss)
            self.trainer = trainer

        trained_network = trainer.fit(self, train, n_epochs=n_epochs, valid=valid, test=test,
                                      augmentation=augmentation, extra_metadata=extra_metadata,
                                      checkpoint=checkpoint, checkpoint_all=checkpoint_all)

        return trained_network
