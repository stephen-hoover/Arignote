"""File interactions related to the Kaggle National Data Science Bowl competition.
"""
from __future__ import division, print_function

from datetime import datetime
import errno
from glob import glob
import gzip
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import theano
import theano.tensor as T


from ..util import netlog
log = netlog.setup_logging("nnets_files", level="INFO")


if sys.version_info.major == 3:
    pickle_load_kwargs = {"encoding": "latin1"}
else:
    pickle_load_kwargs = {}


# Default location of our data.
DATA_DIR = os.path.join(os.getenv("HOME"), "kaggle", "plankton", "competition_data")


class CheckpointError(IOError):
    pass


def get_file_type(fname):
    """Attempts to infer the file type by reading the extension.

    **Returns**

    String, one of ["pkl", "hdf", "csv"]

    **Raises**

    `TypeError` if the file type is unrecognized.
    """
    tokens = fname.split(".")
    for token in tokens[::-1]:
        if token in ["pkl"]:
            return "pkl"
        elif token in ["h5", "hdf"]:
            return "hdf"
        elif token in ["csv"]:
            return "csv"
    else:
        raise TypeError("Unrecognized file type: \"{}\".".format(fname))


def tolerant_makedirs(dirname):
    """This is a wrapper around os.makedirs which will quietly continue without doing anything
    if the specified `dirname` is actually a filename or is an existing directory.
    """
    try:
        os.makedirs(dirname)
    except (IOError, OSError) as e:
        # We get this if the directory exists already, or there's only a filename.
        if e.errno in [errno.ENOENT, errno.EEXIST]:
            pass
        else:
            raise


def read_pickle(fname, **kwargs):
    """Wrapper for pickle.load which will open and close your file for you.
        If the filename ends with "gz" or "gzip" we'll try to read the pickle
        as a gzip file.
    """
    opener = gzip.open if fname.endswith("gz") else open
    pickle_kwargs = pickle_load_kwargs.copy()
    pickle_kwargs.update(kwargs)

    with opener(fname, "rb") as fin:
        contents = pickle.load(fin, **pickle_kwargs)

    return contents


def parse_checkpoint(checkpoint):
    """ Figure out if we want to checkpoint to a single file or to a series of files.
    Assign a default base name if we only got a directory.

    **Returns**

    A 2-tuple of (directory name [str], file name or basename [str])
    """
    if checkpoint is None:
        return None, None
    checkpoint_dir = os.path.dirname(checkpoint)
    checkpoint_fname = os.path.basename(checkpoint)
    if not checkpoint_fname:
        checkpoint_fname = "{}_checkpoint".format(datetime.strftime(datetime.now(), "%Y%M%d"))
    checkpoint_fname = checkpoint_fname.split(".")[0]  # Remove filename extensions.

    return checkpoint_dir, checkpoint_fname


def label_test_images(scorefile, output_file, means, stds, img_scale=63, seed=42,
                      batch_size=128, data_dir=None):

    if data_dir is None:
        data_dir = DATA_DIR
    if not os.path.isdir(os.path.join(data_dir, "test")):
        raise IOError("You must have the data in a directory named {}.".
                          format(os.path.join(data_dir, "test")))

    if scorefile.endswith(".gz"):
        compression = "gzip"
    else:
        compression = None
    scores = pd.read_csv(scorefile, compression=compression, index_col="image")
    top_one = scores.idxmax(axis=1)  # A Series with the name of the most probable class

    # Use a pre-stored pickle of class names to create the output file header.
    # Using the pickle is a hedge against new directories appearing or the class order changing.
    class_fname = os.path.join(data_dir, "plankton_classes.pkl")
    with open(class_fname, "rb") as fin:
        class_list = pickle.load(fin)
    class_dict = dict(zip(class_list, np.arange(len(class_list))))

    def process_func(image, img_scale=img_scale, image_means=means, image_stds=stds):
        return process_images([image], img_scale, means=image_means, stds=image_stds)[0]

    img_x, img_y = [], []
    data_dir = os.path.join(data_dir, "test")
    print("Reading and processing {} test images.".format(top_one.shape[0]))
    for fname, label in top_one.iteritems():
        img_y.append(class_dict[label])
        img_x.append(process_func(imread(os.path.join(data_dir, fname), as_grey=True)))

    img_x = np.asarray(img_x, dtype="float32")
    img_y = np.asarray(img_y, dtype="float32")

    log.info("Shuffling images...")
    rand = np.random.RandomState(seed=seed)
    i_shuffle = stratified_shuffle(img_y, batch_size=batch_size, rng=rand)
    img_x = img_x[i_shuffle]
    img_y = img_y[i_shuffle]

    log.info("Writing images to {} .".format(output_file))
    tolerant_makedirs(os.path.split(output_file)[0])
    with gzip.open(output_file, "wb") as fout:
        pickle.dump((img_x, img_y, means.astype("float32"), stds.astype("float32")),
                    fout, protocol=2)


def score_and_save_test_data(classifier, output_filename, process_func,
                             batch_size=128, data_dir=None, overwrite=False):
    """Function which uses a trained classifier to find probabilities of membership for
    each of the plankton test images, and writes the probabilities to file.

    **Parameters**

    * `classifier` <object>
        An object with a `predict_proba` function.
    * `output_filename` <string>
        Write the class probabilities to this file.
    * `process_func` <function>
        Calling this function on a batch of images must return a batch of images
        processed in exactly the same way as images used during training.

    **Optional Parameters**

    * `batch_size` <int|128>
        Score this many images at once.
    * `data_dir` <string|None>
        Look for the "test" directory at this base directory.
    * `overwrite` <bool|False>
        If False, this function will raise an IOError if the `output_filename` exists.
    """
    # Don't overwrite an existing score file by accident.
    if os.path.exists(output_filename) and not overwrite:
        raise IOError("The file {} already exists. Either choose a different output "
                      "or set `overwrite = True`.".format(output_filename))

    if data_dir is None:
        data_dir = DATA_DIR
    if not os.path.isdir(os.path.join(data_dir, "test")):
        raise IOError("You must have the plankton test images in a directory named {}.".
                          format(os.path.join(data_dir, "test")))

    # Use a pre-stored pickle of class names to create the output file header.
    # Using the pickle is a hedge against new directories appearing or the class order changing.
    class_fname = os.path.join(data_dir, "plankton_classes.pkl")
    with open(class_fname, "rb") as fin:
        class_list = pickle.load(fin)

    opener = gzip.open if output_filename.endswith("gz") else open

    # Read and score all test files.
    all_filenames = np.sort(glob(os.path.join(data_dir, "test", "*.jpg")))
    print("Scoring {} test images.".format(len(all_filenames)))
    with opener(output_filename, "w") as fout:
        fout.write("image,{}\n".format(",".join(class_list)))  # Write the header.
        images, names = [], []
        for i_file, fname in enumerate(all_filenames):
            # Read each image one at a time, score it, and write to the output file.
            images.append(imread(fname, as_grey=True))
            names.append(fname)

            # Process and write a batch of images.
            if len(images) == batch_size:
                images = process_func(images)
                probs = classifier.predict_proba(images.astype(theano.config.floatX))
                for name, prob in zip(names, probs):
                    fout.write("{},{}\n".format(os.path.split(name)[1], ",".join(map(str, prob))))
                images, names = [], []

            if i_file % int(len(all_filenames)/10) == 0:
                print("Wrote {:.0%} of test images.".format((i_file + 1) / len(all_filenames)))

        # Write remaining images.
        if images:
            images = process_func(images)
            probs = classifier.predict_proba(images.astype(theano.config.floatX))
            for name, prob in zip(names, probs):
                fout.write("{},{}\n".format(os.path.split(name)[1], ",".join(map(str, prob))))

    print("Completed processing images. Scores written to {} .".format(output_filename))


def ensemble_scored_files(filenames, output_filename):
    """Take a list of filenames or a string with a wildcard pattern. Assume that each
    is in a valid submission format as written by `score_and_save_test_data`.
    Read the entries, average the probabilities in each row, and re-write the output to disk.
    """
    if not isinstance(filenames, list):
        # Assume it's a string.
        filenames = glob(filenames)

    # Read in probabilities from each file and keep a running sum.
    summed_probs = None
    for fname in filenames:
        log.info("Reading image scores from {} .".format(fname))
        compression = "gzip" if fname.endswith("gz") else None

        these_probs = pd.read_csv(fname, index_col=0, compression=compression)

        if summed_probs is None:
            summed_probs = these_probs
        else:
            summed_probs += these_probs
        these_probs = None  # Don't need to keep this around, and it might be large.

    # Average probabilities by dividing by the number of files.
    summed_probs /= len(filenames)
    log.info("Finished averaging scores.")

    # Check if we need to compress the output. Always write the pandas csv uncompressed.
    if output_filename.endswith(".gz"):
        zip_file = True
        output_filename = output_filename[:-3]
        log.info("I'll write to an uncompressed file, then compress afterwards.")
    else:
        zip_file = False

    # File output.
    log.info("Writing ensembeled model to {} .".format(output_filename))
    summed_probs.to_csv(output_filename, index=True, header=True)

    # Compress the csv if requested, and remove the uncompressed file.
    if zip_file:
        log.info("Compressing file.")
        zipped_fname = output_filename + ".gz"
        with open(output_filename, "rb") as f_in:
            with gzip.open(zipped_fname, "wb") as f_out:
                f_out.writelines(f_in)
        os.unlink(output_filename)
        log.info("Ensembeled model written to {} .".format(zipped_fname))


def checkpoint_write(net, trainer, filename, extra_metadata=None):
    """Checkpoint model training. Stores everything necessary to resume later (except for data).
    """
    tolerant_makedirs(os.path.split(filename)[0])

    if filename.endswith("gz") or filename.endswith("gzip"):
        opener = gzip.open
    else:
        opener = open

    with opener(filename, "wb") as fout:
        # Store metadata about the model type and the model parameters.
        log.debug("Checkpointing to \"{}\".".format(filename))
        metadata = dict(model_store_version="3.0",
                        model_name=getattr(net, "name", None),
                        model_type=type(net),
                        log=log.debug_global,
                        utc_time=time.asctime(time.gmtime()))

        if extra_metadata is not None:
            metadata.update(extra_metadata)

        pickle.dump(metadata, fout, protocol=2)
        pickle.dump(net, fout, protocol=2)  # Protocol 2 for Python 2 compatibility.


def checkpoint_read(filename, get_metadata=False):
    """Reads a model checkpoint of V3.0 or higher.
    """
    if filename.endswith("gz") or filename.endswith("gzip"):
        opener = gzip.open
    else:
        opener = open

    with opener(filename, "rb") as fin:
        # Recover metadata, assumed to be a dictionary.
        metadata = pickle.load(fin, **pickle_load_kwargs)

        if not isinstance(metadata, dict) or float(metadata.get("model_store_version", "0")) < 3:
            log.warning("Can't read file {} as a checkpoint.".format(filename))
            raise CheckpointError("This file does not look like a V3.0 or better checkpoint.")

        net = pickle.load(fin, **pickle_load_kwargs)

    if get_metadata:
        return net, metadata
    else:
        return net


def save_model(model, filename, extra_metadata=None):
    """Saves the parameters in an input model to a gzipped pickle file.

    **Parameters**

    * `model`: An object with a `params` attribute which is a list of Theano shared variables.
    * `filename` <str>: Location to save the model.
    """
    tolerant_makedirs(os.path.split(filename)[0])

    with gzip.open(filename, "wb") as fout:
        # Store metadata about the model type and the model parameters.
        metadata = dict(model_store_version="2.0",
                        model_name=getattr(model, "name", None),
                        model_type=type(model),
                        params=[(p.name, p.type, p.ndim) for p in model.params],
                        utc_time=time.asctime(time.gmtime()),
                        init_params={})
        if hasattr(model, "get_init_params"):
            metadata["init_params"] = model.get_init_params()

        if extra_metadata is not None:
            metadata.update(extra_metadata)

        # Construct a dictionary for pickling.
        # Preserve the ordering of the parameters, but store both the name of the layer
        # to which the parameters belong and the names of the parameters.
        params = []
        for layer in model.layers_train:
            params.append((layer.name,
                           [(par.name, par.get_value(borrow=True)) for par in layer.params]))
        update_rules = [(lyr.name, lyr.param_update_rules) for lyr in model.layers_train]
        pickle.dump({"metadata": metadata,
                     "params": params,
                     "param_update_rules": update_rules},
                    fout, protocol=2)  # Protocol 2 for Python 2 compatibility.


def create_class_list(dir=None, filename_to_write="plankton_classes"):
    if dir is None:
        dir = DATA_DIR

    # Read all of the directories in the "train" folder.
    all_dirs = glob(os.path.join(dir, "train", "*"))
    if len(all_dirs) != 121:
        log.warning("I expected 121 directories, but found {}.".format(len(all_dirs)))

    # Split off the last directory name.
    all_dirs = [os.path.split(name)[1] for name in all_dirs]

    # Write to disk.
    fname = os.path.join(dir, "{}.pkl".format(filename_to_write))
    with open(fname, "wb") as fin:
        pickle.dump(all_dirs, fin, protocol=2)  # Protocol 2 for Python 2 compatibility

    log.info("Wrote {} directory names to {}.".format(len(all_dirs), fname))
    return all_dirs


def restore_model(filename, model=None, input=None, **kwargs):
    """Reads both V1 and V2 model stores.
    """
    try:
        return restore_model_v2(filename, model=model, input=input, **kwargs)
    except TypeError:
        return restore_model_v1(filename, model=model, input=input, **kwargs)


def restore_model_v1(filename, model=None, input=None, skip_hl=False):
    """Reads V1 model stores.
    Loads model parameters from the given filename. If a model is supplied,
    we copy the stored parameters into the input model. Otherwise, create a fresh
    model using supplied inputs.

    **Parameters**

    * `filename` <str>: Location of a gizipped pickle file containing model parameters.

    **Optional Parameters**

    * `model` <None>: An object with a `params` attribute which is a list of Theano shared variables.
    * `input` <None>: Theano symbolic tensor, to be supplied to a newly-created model as
        inputs. Ignored if `model` is not None.

    **Returns**

    A 2-tuple of the new model object and the metadata dictionary stored in the given file.

    **Modifies**

    If supplied, the input `model` object will be altered so that its parameters contain the values
    stored in the specified file.

    **Raises**

    `IOError` if the supplied pickle file has an unexpected format.
    `TypeError` if the parameters in the file don't fit into the supplied `model` object.
    `TypeError` if neither `model` nor `input` are supplied.
    """
    if model is None and input is None:
        input = T.matrix("x")  # Data, presented as rasterized images

    with gzip.open(filename, "rb") as fin:
        # Recover metadata, assumed to be a dictionary.
        metadata = pickle.load(fin, **pickle_load_kwargs)

        # If we didn't get an input model, create it.
        if model is None:
            full_name = str(metadata["model_type"]).split("'")[1]
            module_name = full_name.split(".")[:-1]
            class_name = full_name.split(".")[-1]
            module = __import__(".".join(module_name), fromlist=module_name[:-1])
            model = getattr(module, class_name)(**metadata["init_params"])
            model.compile(input=input)

        # Verify that the model we have fits the parameters in the model store.
        if len(metadata["params"]) != len(model.params):
            raise TypeError("The input model contains {} parameters, but this model store "
                            "contains {} parameters. Model store for model \"{}\" of type {}, "
                            "written on {}.".format(len(model.params), len(metadata["params"]),
                                                    metadata["model_name"], metadata["model_type"],
                                                    metadata["utc_time"]))

        # Put the stored values of the parameters in the model.
        for param in model.params:
            if skip_hl and param.name.endswith("hl"):
                log.warning("Not restoring parameter {} .".format(param.name))
                continue
            else:
                param.set_value(pickle.load(fin, **pickle_load_kwargs), borrow=True)

    return model, metadata


def restore_model_v2(filename, model=None, input=None, skip_hl=False):
    """Reads V2 model stores.
    Loads model parameters from the given filename. If a model is supplied,
    we copy the stored parameters into the input model. Otherwise, create a fresh
    model using supplied inputs.

    **Parameters**

    * `filename` <str>: Location of a gizipped pickle file containing model parameters.

    **Optional Parameters**

    * `model` <None>: An object with a `params` attribute which is a list of Theano shared variables.
    * `input` <None>: Theano symbolic tensor, to be supplied to a newly-created model as
        inputs. Ignored if `model` is not None.

    **Returns**

    A 2-tuple of the new model object and the metadata dictionary stored in the given file.

    **Modifies**

    If supplied, the input `model` object will be altered so that its parameters contain the values
    stored in the specified file.

    **Raises**

    `IOError` if the supplied pickle file has an unexpected format.
    `TypeError` if the parameters in the file don't fit into the supplied `model` object.
    `TypeError` if neither `model` nor `input` are supplied.
    """
    if model is None and input is None:
        input = T.matrix("x")  # Data, presented as rasterized images

    with gzip.open(filename, "rb") as fin:
        # Recover metadata, assumed to be a dictionary.
        model_data = pickle.load(fin, **pickle_load_kwargs)

    if not ("metadata" in model_data and float(model_data["metadata"]["model_store_version"]) >= 2):
        raise TypeError("Use this function to read V2.0 and greater model stores.")

    metadata = model_data["metadata"]
    params = model_data["params"]

    # If we didn't get an input model, create it.
    if model is None:
        full_name = str(metadata["model_type"]).split("'")[1]
        module_name = full_name.split(".")[:-1]
        class_name = full_name.split(".")[-1]
        module = __import__(".".join(module_name), fromlist=module_name[:-1])
        model = getattr(module, class_name)(**metadata["init_params"])
        model.compile(input=input)

    # Verify that the model we have fits the parameters in the model store.
    if len(params) != len(model.layers_train):
        raise TypeError("The input model contains {} layers, but this model store "
                        "contains {} layers. Model store for model \"{}\" of type {}, "
                        "written on {}.".format(len(model.layers_train), len(params),
                                                metadata["model_name"], metadata["model_type"],
                                                metadata["utc_time"]))

    # Put the stored values of the parameters in the model.
    for layer, stored_params in zip(model.layers_train, params):
        for param, (par_name, value) in zip(layer.params, stored_params[1]):
            param.set_value(value, borrow=True)

    return model, metadata


def format_metadata(metadata):
    metacopy = metadata.copy()
    for key in ["all_train_cost", "all_valid_loss", "all_test_nll", "all_test_error", "train_loss",
                "valid_loss", "test_loss", "train_valid_loss"]:
        if key in metacopy:
            del metacopy[key]
    metastr = """{}""".format(str(metacopy))

    return metastr
