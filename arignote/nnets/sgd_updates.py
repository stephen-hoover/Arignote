"""
This module contains parameter update functions for stochastic gradient descent optimization.
"""
from __future__ import division, print_function

import numpy as np
import theano
import theano.tensor as T

from ..util import netlog


log = netlog.setup_logging("nnets_sgd_updates", level="INFO")


class Rule(object):
    """
    # Subtract 1e-4 every epoch until reaching 1e-3.
    Rule("fixed", decrease_by=1e-4, interval=1, final_value=1e-3)

    # Multiply by 0.5 every 10 epochs until reaching a value of 1e-4.
    Rule("fixed", multiply_by=0.5, interval=10, final_value=1e-4)

    # Multiply by 0.5 at epochs 15, 23, 30, and every 5 epochs after.
    Rule("fixed", multiply_by=0.5, schedule=[15, 23, 30], interval=5)

    # Divide by 10 every time we go 5 epochs without an improvement.
    Rule("stalled", multiply_by=0.1, interval=5)
    """
    valid_rules = ["const", "constant", "fixed", "stalled"]

    def __init__(self, rule, initial_value,
                 final_value=None, decrease_by=0., multiply_by=1.,
                 interval=None, schedule=None):

        # Store initialization parameters.
        self.rule = rule.lower()
        if rule not in self.valid_rules:
            raise ValueError("Allowed rules are {}.".format(self.valid_rules))

        self.initial_value = initial_value
        self.final_value = final_value
        self.decrease_by = decrease_by
        self.multiply_by = multiply_by
        self.interval = interval
        self.schedule = schedule
        if self.schedule:
            # If we have a schedule, flip it around so that we pick the first entries first.
            self.schedule = np.sort(self.schedule).tolist()[::-1]

        # Remember where we've been.
        self.last_epoch = 0  # Last time that we adjusted the learning rate.
        self.best_valid = np.inf
        self.epochs_since_best = 0
        self.update_epochs = []  # Add epochs where we've made an update.

    def __str__(self):
        """Describe the contents of this Rule for humans.
        """
        display = "Rule: "
        if self.rule in ["const", "constant"]:
            display += "held constant at {}".format(self.initial_value)
        else:
            # First describe when we change a value.
            if self.rule in ["fixed"]:
                display += "change on a fixed schedule: "

                time_str = []
                if self.schedule is not None:
                    time_str.append("at validations {}".format(self.schedule))
                if self.interval is not None:
                    time_str.append("every {} validations".format(self.interval))
                time_str = " and then ".join(time_str)
            elif self.rule in ["stalled"]:
                display += "when no improvement after "

                time_str = []
                if self.schedule is not None:
                    time_str.append("{} consecutive checks".format(self.schedule))
                if self.interval is not None:
                    time_str.append("{} consecutive checks".format(self.interval))
                time_str = " and then ".join(time_str)
            else:
                raise ValueError("Unrecognized rule: {}".format(self.rule))

            # Now describe how we change the value when we do change it.
            change = []
            if self.decrease_by > 0:
                change.append("subtract {}".format(self.decrease_by))
            elif self.decrease_by < 0:
                change.append("add {}".format(-self.decrease_by))
            if self.multiply_by != 1:
                change.append("multiply by {}".format(self.multiply_by))
            change = " and then ".join(change)

            # Join everything together.
            display = "{}{}, {}".format(display, time_str, change)

        return display

    def is_update_time(self, epoch, loss):
        """Checks if it's time to do an update and returns True if so.
        If we're doing dynamic updates, this involves modifying our record of the best
        """
        if self.rule in ["const", "constant"]:
            return False  # Never update with rule "constant".
        elif self.rule in ["stalled"]:
            if loss < self.best_valid:
                self.last_epoch = epoch
                self.best_valid = loss
            sch_adj = self.last_epoch  # Subtract from this epoch when checking on the schedule.
        elif self.rule in ["fixed"]:
            sch_adj = 0
        else:
            raise ValueError("Unrecognized timing rule: {}".format(self.rule))

        # First figure out if we need to do the update.
        do_update = False
        if self.schedule:
            # If there's any items left in the schedule, ignore the "interval".
            if (epoch - sch_adj) == self.schedule[-1]:
                self.schedule.pop()  # We've completed this part of the schedule now.
                do_update = True
        elif self.interval and (epoch - self.last_epoch) == self.interval:
            do_update = True

        return do_update

    def update(self, value, epoch, loss):
        """Modifies the value according to the rule.
        """
        do_update = self.is_update_time(epoch, loss)
        if do_update:
            prev_value = value.get_value()
            value.set_value(self.multiply_by * (prev_value - self.decrease_by))

            # Make sure the value can't pass its limits.
            if self.final_value is not None:
                func = np.min if (value.get_value() - prev_value > 0) else np.max
                value.set_value(func([value.get_value(), self.final_value]))
            self.last_epoch = epoch

            if value.get_value() == prev_value:
                # Make a note if we didn't actually change anything.
                do_update = False

        if do_update:
            self.update_epochs.append(epoch)

        return do_update

    def reset(self, value):
        value.set_value(self.initial_value)

        self.last_epoch = 0  # Last time that we adjusted the learning rate.
        self.best_valid = np.inf
        self.epochs_since_best = 0
        self.update_epochs = []  # Add epochs where we've made an update.


class SGD(object):
    def __init__(self, sgd_type, lr_rule=None, momentum_rule=None,
                 max_grad_norm=None):
        """Pick the appropriate update rule.

        **Parameters**

        * `sgd_type` <string> : "nag", "rmsprop", etc.

        * `lr_rule` <Rule|None> : A rule which determines how to modify the learning rate.
            If None, the learning rate will be constant.

        **Optional Parameters**

        * `momentum_rule` <Rule|None> : The rule that determines how to modify the
            momentum as training progresses. If None, momentum will be held constant.

        * `max_grad_norm` <float|None> : If the total norm of the gradient for any
            parameter array exceeds this value, it will be scaled back to equal this value.
            Disabled if None.
        """
        if isinstance(lr_rule, dict):
            lr_rule = Rule(**lr_rule)
        if isinstance(momentum_rule, dict):
            momentum_rule = Rule(**momentum_rule)

        if lr_rule is None:
            lr_rule = Rule("constant", 1.0)
        if not isinstance(lr_rule, Rule):
            raise TypeError("The `lr_rule` must be a Rule.")
        if momentum_rule is None:
            momentum_rule = Rule("constant", 0.95)
        if not isinstance(momentum_rule, Rule):
            raise TypeError("The `momentum_rule` must be a Rule.")

        self.lr_rule = lr_rule
        self.momentum_rule = momentum_rule
        self.update_params = []  # Store here the changeable update parameters, e.g. momentum.
        self.stored_update_params = None  # Anything restored from pickling will go here.
        self.sgd_type = sgd_type
        self.max_grad_norm = max_grad_norm

        self._setup()

    def _setup(self):
        self.max_sqr_grad_norm = self._init_max_grad_norm(self.max_grad_norm)
        self.learning_rate, self.momentum = self._init_lr_momentum(self.lr_rule.initial_value,
                                                                   self.momentum_rule.initial_value)
        self.update_creator, self.update_kwargs = self._init_update_creator(self.sgd_type)

    def _init_lr_momentum(self, lr_val, momentum_val):
        """Create shared variables for learning rate and momentum."""
        # Create Theano shared variables for the learning rate and momentum.
        # If they're shared variables, we can alter them here and have those
        # changes show up in any Theano functions that they're a part of.
        learning_rate = T.sharedvar.ScalarSharedVariable("learning_rate", T.TensorType(theano.config.floatX, []),
                                                         np.cast[theano.config.floatX](lr_val),
                                                         strict=False, allow_downcast=True)
        momentum = T.sharedvar.ScalarSharedVariable("momentum", T.TensorType(theano.config.floatX, []),
                                                    np.cast[theano.config.floatX](momentum_val),
                                                    strict=False, allow_downcast=True)
        return learning_rate, momentum

    def _init_max_grad_norm(self, max_grad_norm):
        """Create a shared variable for the square of the max gradient norm. Note that the
        input to this function is the max norm, and the shared variable is the /square/.
        """
        if max_grad_norm is None:
            max_sqr_grad_norm = None
        else:
            max_sqr_grad_norm = T.sharedvar.ScalarSharedVariable("max_sqr_grad_norm",
                                            T.TensorType(theano.config.floatX, []),
                                            np.cast[theano.config.floatX](max_grad_norm ** 2),
                                            strict=False, allow_downcast=True)
        return max_sqr_grad_norm

    def _init_update_creator(self, sgd_type):
        """Grab the function which will process the SGD updates, and find which arguments it takes.

        **Returns**

        2-tuple of (function, dictionary), where the dictionary has the shared parameters which
        control the updates (e.g. learning rate) keyed by name.

        **Modifies**

        None
        """
        update_kwargs = dict(learning_rate=self.learning_rate)
        if sgd_type in ["nag", "momentum"]:
            #log.info("Using Nesterov accelerated gradient descent. LR = {}; momentum = {}".
            #                 format(self.learning_rate.get_value(), self.momentum.get_value()))
            update_creator = self.gradient_updates_momentum
            update_kwargs.update(dict(momentum=self.momentum, nesterov=True))
        elif sgd_type == "rmsprop":
            #log.info("Using RMSprop gradient descent. LR = {}; rho = {}".
            #                 format(self.learning_rate.get_value(), self.momentum.get_value()))
            update_creator = self.gradient_updates_rmsprop
            update_kwargs.update(dict(rho=self.momentum))
        elif sgd_type == "adagrad":
            #log.info("Using adagrad updates for gradient descent. LR = {}".
            #                 format(self.learning_rate.get_value()))
            update_creator = self.gradient_updates_adagrad
        elif sgd_type == "adadelta":
            #log.info("Using adadelta updates for gradient descent. LR = {}; rho = {}".
            #                 format(self.learning_rate.get_value(), self.momentum.get_value()))
            update_creator = self.gradient_updates_adadelta
            update_kwargs.update(dict(rho=self.momentum))
        elif sgd_type == "sgd":
            #log.info("Using bog-standard stocastic gradient descent.")
            update_creator = self.gradient_updates_sgd
        else:
            raise ValueError("Unrecognized SGD algorithm: {}".format(sgd_type))

        return update_creator, update_kwargs

    def __getstate__(self):
        state = self.__dict__.copy()
        state["learning_rate"] = state["learning_rate"].get_value(borrow=True)
        state["momentum"] = state["momentum"].get_value(borrow=True)
        max_sqr_grad_norm = state.pop("max_sqr_grad_norm")
        if max_sqr_grad_norm is not None:
            state["_max_grad_norm"] = np.sqrt(max_sqr_grad_norm.get_value(borrow=True))
        else:
            state["_max_grad_norm"] = None

        # The "update_params" are a list of dictionaries of e.g. current momenta.
        # Convert all of the shared variables to arrays for storage.
        state["stored_update_params"] = [{k: v.get_value(borrow=True) for k, v in p_dict.items()}
                                         for p_dict in self.update_params]
        state["update_params"] = []
        del state["update_creator"]
        del state["update_kwargs"]

        return state

    def __setstate__(self, state):
        self.learning_rate, self.momentum = self._init_lr_momentum(state.pop("learning_rate"),
                                                                   state.pop("momentum"))
        self.max_sqr_grad_norm = self._init_max_grad_norm(state.pop("_max_grad_norm"))
        self.__dict__.update(state)

        self.update_creator, self.update_kwargs = self._init_update_creator(self.sgd_type)

    def reset(self):
        self.lr_rule.reset(self.learning_rate)
        self.momentum_rule.reset(self.momentum)
        for param_dict in self.update_params:
            for value in param_dict.values():
                value.set_value(0 * value.get_value())

    def __str__(self):
        description = "SGD with {} updates. Learning rate = {}; momentum = {}.".\
                format(self.sgd_type, self.learning_rate.get_value(), self.momentum.get_value())
        description += "\n\tUpdate learning rate according to {}".format(str(self.lr_rule))
        description += "\n\tUpdate momentum according to {}".format(str(self.momentum_rule))

        return description

    def update_lr(self, epoch, loss):
        lr_updated = self.lr_rule.update(self.learning_rate, epoch=epoch, loss=loss)
        momentum_updated = self.momentum_rule.update(self.momentum, epoch=epoch, loss=loss)

        return lr_updated or momentum_updated

    def get_updates(self, cost, params, param_rules=None, reset=False):
        """Return update rules for each input parameter.
        If this object has previously used SGD parameter states (e.g. if it's been restored
        from a pickled state), then initialize momenta from those.

        **Parameters**

        * `cost` <Theano symbolic function> : Take the gradient of this cost with respect
            to each parameter to determine how to update the parameter.

        * `params` <list of Theano shared variables> : Parameters to update.

        **Optional Parameters**

        * `param_rules` <list|None> : List of dictionaries, ordered to match the `params`.

        * `reset` <bool|False> : If True, ignore any stored update parameter and initialize
            everything fresh.
        """
        if param_rules is None:
            param_rules = [{} for ii in range(len(params))]
        stored_params = self.stored_update_params
        if stored_params is None or reset:
            stored_params = [{} for ii in range(len(params))]
        else:
            log.debug("Restoring SGD parameters from saved state.")
            if len(params) != len(stored_params):
                raise ValueError("Number of input parameters does not match number of"
                                 "stored SGD update parameters.")

        updates = []
        for param, param_rule, prev_param in zip(params, param_rules, stored_params):
            # Check if we should just skip updating this parameter completely.
            if param_rule.get("fixed", False):
                log.debug("Parameter {} is fixed.".format(param.name))
                continue

            creator_kwargs = prev_param.copy()
            creator_kwargs.update(self.update_kwargs)

            # Assume that the update creator function returns a list of update tuples.
            this_ud, ud_params = self.update_creator(cost=cost, param=param, **creator_kwargs)
            updates.extend(this_ud)
            self.update_params.append(ud_params)

            # We may want to modify these updates (for example, to clip weights based on a
            # column max norm).
            self.post_update_updates(updates, param_rule)

        return updates

    def post_update_updates(self, updates, param_rules):
        """Recognized rules for modifying the last update:
            "maxnorm" : Float. Any columns of > 1D parameters with norm greater than this value
                will be clipped to this value.

        **Parameters**

        * `updates` <list> : List of 2-tuple (param, new param) updates.
        * `param_rules` <dict> : Dictionary of rules to apply to the last update.

        ***Returns**

        None

        **Modifies**

        The input `updates` list may have its last element replaced.
        """
        epsilon = 1e-7
        last_update = updates[-1]
        if param_rules.get("maxnorm", None) is not None:
            # Maxnorm code taken from "theanet.neuralnet",
            # https://github.com/rakeshvar/theanet/blob/master/theanet/neuralnet.py
            if last_update[0].get_value(borrow=True).ndim == 2:
                col_norms = T.sqrt(T.sum(T.sqr(last_update[1]), axis=0))
                desired_norms = T.clip(col_norms, 0, param_rules["maxnorm"])
                scale = (epsilon + desired_norms) / (epsilon + col_norms)
                updates[-1] = (last_update[0], scale * last_update[1])

    def grad_restrict(self, cost, wrt, **kwargs):
        """Scales back the gradient of a parameter if the total norm is too large.
        This was recommended by Ilya Sutskever for RNNs and LSTMs at
        http://yyue.blogspot.ca/2015/01/a-brief-overview-of-deep-learning.html .
        """
        grad = T.grad(cost=cost, wrt=wrt, **kwargs)
        if self.max_sqr_grad_norm is not None:
            epsilon = 1e-7
            grad_norm = T.sum(T.sqr(grad))
            desired_norm = T.clip(grad_norm, 0, self.max_sqr_grad_norm)
            scale = T.sqrt((epsilon + desired_norm) / (epsilon + grad_norm))
            grad = scale * grad

        return grad

    def gradient_updates_sgd(self, cost, param, learning_rate):
        """Standard, boring, stochastic gradient descent with no bells or whistles.
        """
        return [(param, param - learning_rate * self.grad_restrict(cost, param))], {}

    def gradient_updates_rmsprop(self, cost, param, learning_rate, rho, acc=None, epsilon=1e-6):
        """Compute SGD updates using RMSprop. Function taken from
        https://github.com/Newmu/Theano-Tutorials/blob/master/5_convolutional_net.py

        The RMSprop algorithm accumulates a measure of the RMS of previously seen
        gradients, and divides the gradient at this location by the RMS.
        This scaling dynamically adjusts the learning rate, allowing for fast learning
        when the gradients are changing slowly, and smaller updates when the cost function
        is changing quickly.

        **Parameters**

        * `cost` <theano.tensor.var.TensorVariable>
                Theano cost function to minimize
        * `param` <theano.tensor.var.TensorVariable>
                Parameter to compute gradient against
        * `learning_rate` <float>
                Base gradient descent learning rate
        * `rho` <float>
                Controls how quickly our accumlated measure of the squared gradient changes.
                The updated squared gradient is equal to a fraction `rho` of the previous squared
                gradient and `1 - rho` of the squared gradient in the current location.
                Must be between 0 and 1.

        * `acc` <ndarray|None>
                Initialize accelerations to this array, if present.
        * `epsilon` <float|1e-6>
                Numerical regularization parameter to prevent us from dividing by an
                especially tiny RMS.

        **Returns**
            updates : list
                List of updates, one for each parameter
        """
        # Don't let `rho` be crazy.
        if hasattr(rho, "get_value"):
            assert 0 < rho.get_value() < 1, "Your rho value must be in the range (0, 1)."
        else:
            assert 0 < rho < 1, "Your rho value must be in the range (0, 1)."
        grad = self.grad_restrict(cost=cost, wrt=param)
        updates = []

        if acc is None:
            # Initialize to zeros.
            acc = param.get_value() * 0.
        acc = theano.shared(acc.astype(theano.config.floatX), broadcastable=param.broadcastable)
        acc_new = rho * acc + (1 - rho) * grad ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        grad = grad / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((param, param - learning_rate * grad))
        return updates, {"acc": acc}

    def gradient_updates_momentum(self, cost, param, learning_rate, momentum,
                                  velocity=None, nesterov=True):
        """Compute updates for gradient descent with momentum.
        Based on an example by Colin Raffel (http://colinraffel.com/) at
        http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb

        For implementation of Nesterov momentum, see http://arxiv.org/pdf/1212.0901v2.pdf
        Thanks to PyLearn (https://github.com/lisa-lab/pylearn2/pull/1030)
        See also http://www.cs.toronto.edu/~fritz/absps/momentum.pdf

        **Parameters**

        * `cost` <theano.tensor.var.TensorVariable>
                Theano cost function to minimize
        * `param` <theano.tensor.var.TensorVariable>
                Parameter to compute gradient against
        * `learning_rate` <float>
                Gradient descent learning rate
        * `momentum` <float>
                Momentum parameter, should be at least 0 (standard gradient descent) and less than 1

        * `velocity` <ndarray|None>
                Initialize the velocity array to these values, else they start at zero.
        * `nesterov` <bool|True>
                Use the Nesterov accelerated gradient decent update rule? (You should.)

        **Returns**
            updates : list
                List of updates, one for each parameter
        """
        # Make sure momentum is a sane value.
        if hasattr(momentum, "get_value"):
            assert 0 <= momentum.get_value() < 1, "Momentum must be between 0 and 1."
        else:
            assert 0 <= momentum < 1, "Momentum must be between 0 and 1."
        # List of update steps for each parameter.
        updates = []

        # Just gradient descent on cost
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0.
        if velocity is None:
            velocity = param.get_value() * 0.
        velocity = theano.shared(velocity.astype(theano.config.floatX),
                                 broadcastable=param.broadcastable)

        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also
        # the new gradient step.
        # Standard momentum uses the paired updates
        # v_t = mu * v_t-1 - lr * grad(cost, param_t-1)
        # param_t = param_t-1 + v_t
        # Here we use Nesterov's accelerated gradient decent algorithm ("Nesterov momentum").
        # http://www.cs.toronto.edu/~fritz/absps/momentum.pdf
        # This uses the modified update rules
        # v_t = mu * v_t-1 - lr * grad(cost, param_t-1 + mu * v_t-1)
        # param_t = param_t-1 + v_t
        # Alternately, this is equivalent to
        # v_t = mu * v_t-1 - lr * grad(cost, param_t-1)
        # param_t = param_t-1 + mu * v_t - lr * grad(cost, param_t-1)

        # First update the velocity.
        updates.append((velocity, momentum * velocity - learning_rate * self.grad_restrict(cost, param)))
        # Now update the parameters.
        if nesterov:
            updates.append((param, param + momentum * velocity - learning_rate * self.grad_restrict(cost, param)))
        else:
            updates.append((param, param + velocity))
        return updates, {"velocity": velocity}

    def gradient_updates_adagrad(self, cost, param, learning_rate=1.0,
                                 accumulator=None, epsilon=1e-6):
        """
        Epsilon is not included in the typical formula,
        See "Notes on AdaGrad" by Chris Dyer for more info.

        This implementation from lasagne:
        https://github.com/benanne/Lasagne/blob/master/lasagne/updates.py
        """
        grad = self.grad_restrict(cost, param)
        if accumulator is None:
            accumulator = np.zeros(param.get_value().shape, dtype=theano.config.floatX)
        accumulator = theano.shared(accumulator.astype(theano.config.floatX),
                                    broadcastable=param.broadcastable)

        updates = []
        acc_new = accumulator + grad**2
        updates.append((accumulator, acc_new))
        updates.append((param, param - learning_rate * grad / T.sqrt(acc_new + epsilon)))

        return updates, {"accumulator": accumulator}

    def gradient_updates_adadelta(self, cost, param, learning_rate=1.0, rho=0.95,
                                  accumulator=None, delta_acc=None, epsilon=1e-6):
        """
        In the paper, no learning rate is considered (so learning_rate=1.0).
        Probably best to keep it at this value.
        Epsilon is important for the very first update (so the numerator does not become 0).
        rho = 0.95 and epsilon=1e-6 are suggested in the paper and reported to work for
         multiple datasets (MNIST, speech).
        See "Adadelta: an adaptive learning rate method" by Matthew Zeiler for more info.

        This implementation from lasagne:
        https://github.com/benanne/Lasagne/blob/master/lasagne/updates.py
        """
        grad = self.grad_restrict(cost, param)

        if accumulator is None:
            accumulator = np.zeros(param.get_value().shape, dtype=theano.config.floatX)
        accumulator = theano.shared(accumulator.astype(theano.config.floatX),
                                    broadcastable=param.broadcastable)
        if delta_acc is None:
            delta_acc = np.zeros(param.get_value().shape, dtype=theano.config.floatX)
        delta_acc = theano.shared(delta_acc.astype(theano.config.floatX),
                                  broadcastable=param.broadcastable)

        # `accumulator`: accumulate gradient magnitudes
        # `delta_acc`: accumulate update magnitudes (recursive!)

        updates = []
        acc_new = rho * accumulator + (1 - rho) * grad**2
        updates.append((accumulator, acc_new))

        update = grad * T.sqrt(delta_acc + epsilon) / T.sqrt(acc_new + epsilon)  # Use the 'old' acc_delta here
        updates.append((param, param - learning_rate * update))

        delta_acc_new = rho * delta_acc + (1 - rho) * update**2
        updates.append((delta_acc, delta_acc_new))

        return updates, {"accumulator": accumulator, "delta_acc": delta_acc}
