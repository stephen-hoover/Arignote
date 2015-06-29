"""
This module reads and iterates over data, making it available for training.
"""
from __future__ import division

import abc
import six
import threading

import numpy as np
import pandas as pd
import theano

from ..util import misc
from ..util import netlog
from ..data import files
log = netlog.setup_logging("data_readers", level="INFO")


def to_data_object(data, batch_size=128, **kwargs):
    """Wrap the input in a Data object. If it has length 1, assume that it's a 1-tuple containing
    an array of features. If length 2, assume the second element is the labels. Extra keyword
    arguments will be passed to the `Data` constructor. The keyword arguments will be ignored
    if `data` is already a Data object or is None."""
    if data is None or isinstance(data, Data):
        obj = data
    else:
        labels = kwargs.pop("labels", None)
        if len(data) == 1:
            features = data[0]
        elif len(data) == 2:
            features, labels = data
        else:
            features = data

        obj = Data(features, labels, batch_size=batch_size, **kwargs)

    return obj


def to_data_partitions(train, valid=0, test=0, batch_size=128, **kwargs):
    """Wrap the input in a DataWithHoldoutPartitions object.
    If it has length 1, assume that it's a 1-tuple containing
    an array of features. If length 2, assume the second element is the labels. Extra keyword
    arguments will be passed to the `Data` constructor. The keyword arguments will be ignored
    if `data` is already a DataWithHoldoutPartitions object."""
    if isinstance(train, DataWithHoldoutParitions):
        output = train.train, train.valid, train.test
    elif isinstance(train, Data):
        if valid == 0:
            valid = None
        if test == 0:
            test = None
        if ((valid is not None and not isinstance(valid, Data))
                or (test is not None and not isinstance(test, Data))):
            raise TypeError("If inputting training data as a `Data` object, validation and"
                            "test sets must also be presented as `Data` objects.")
        output = (train, valid, test)
    else:
        train_labels = None
        if len(train) == 1:
            features = train[0]
        elif len(train) == 2:
            features, train_labels = train
        else:
            features = train

        valid_frac, test_frac = 0, 0
        if misc.is_floatlike(valid):
            valid_frac = valid
            valid = None
        else:
            valid = to_data_object(valid, batch_size=batch_size, allow_partial_batch=True, **kwargs)
        if misc.is_floatlike(test):
            test_frac = test
            test = None
        else:
            test = to_data_object(test, batch_size=batch_size, allow_partial_batch=True, **kwargs)

        obj = DataWithHoldoutParitions(features, labels=train_labels, valid_frac=valid_frac,
                                       test_frac=test_frac, batch_size=batch_size, **kwargs)

        if valid is None:
            valid = obj.valid
        if test is None:
            test = obj.test
        output = obj.train, valid, test

    return output


def threaded_generator(generator, num_cached=10):
    """Wrap a generator in a thread, using a queue to return data.
    Note that due to the Python GIL, this will not allow generators to work while other
    Python code is running. If part of a program releases the GIL, however, this
    wrapper can store up extra items from the generator it wraps.

    Threaded generator implementation due to Jan Schlueter, https://github.com/f0k
    https://github.com/Lasagne/Lasagne/issues/12#issuecomment-59494251
    """
    queue = six.moves.queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()


class Data(object):
    """
    This is the base class for all data iterators suitable for use in training, and can
    be used for simple data iteration.
    """
    def __init__(self, features, labels=None, batch_size=128, alter_features=None,
                 alter_labels=None, start=0, stop=None, allow_partial_batch=False):
        self.batch_size = batch_size
        self.features = features
        self.labels = labels
        self.alter_features = alter_features
        self.alter_labels = alter_labels
        self.start = start
        self.stop = stop
        self.allow_partial_batch = allow_partial_batch

        if self.batch_size is None:
            raise TypeError("Batch size may not be None!")

        self.n_rows = 0
        self._setup()

    def __len__(self):
        stop = self.n_rows if (self.stop is None or self.stop > self.n_rows) else self.stop
        return stop - self.start

    def _setup(self):
        """Execute setup tasks, both input checking and creating derived attributes."""
        # Turn non-Reader data inputs into Readers.
        self.features = get_reader(self.features, labels=False)
        if self.labels is not None:
            self.labels = get_reader(self.labels, labels=True)
        self.n_rows = len(self.features)

        # For alteration, turn None into a do-nothing function.
        if self.alter_features is None:
            self.alter_features = lambda x: x
        if self.alter_labels is None:
            self.alter_labels = lambda x: x

        # Check the inputs.
        if self.labels is not None and len(self.features) != len(self.labels):
            raise ValueError("The features have {} rows, but the labels have {} "
                             "rows.".format(len(self.features), len(self.labels)))

        # Figure out where we're starting each section of the data as a fraction of the whole.
        self.n_epochs = 0

    def iter_epoch(self, num_cached=3):
        for item in threaded_generator(self.iter_epoch_single(), num_cached=num_cached):
            yield item

    def iter_epoch_single(self):
        """Iterate through the data represented by this object.

        **Yields**

        A 2-tuple minibatch of (features, labels) if this object holds labels, else
        a minibatch of features.
        """
        # Set up the feature and label iterators.
        feature_rdr = self.features.iter_epoch(batch_size=self.batch_size,
                                               start=self.start,
                                               stop=self.stop,
                                               start_on_batch=True,
                                               allow_partial=self.allow_partial_batch)

        data = feature_rdr
        if self.labels is not None:
            label_rdr = self.labels.iter_epoch(batch_size=self.batch_size,
                                               start=self.start,
                                               stop=self.stop,
                                               start_on_batch=True,
                                               allow_partial=self.allow_partial_batch)
            data = six.moves.zip(feature_rdr, label_rdr)

        # Iterate over the data.
        for item in data:
            if self.labels is not None:
                yield self.alter_features(item[0]), self.alter_labels(item[1])
            else:
                yield self.alter_features(item)
        self.n_epochs += 1

    def peek(self):
        """Return the first epoch of data."""
        return next(self.iter_epoch_single())


class DataWithHoldoutParitions(object):
    """
    This class partitions input data into three sections: training data, validation data,
    and testing data. It uses rows of input data in the order it finds them.
    The first section of the data will be used for training, the middle section for validation,
    and the last section for testing.

    This object will have a training set as `self.train`, a validation set (if any) as
    `self.valid`, and a test set (if any) as `self.test`.
    """
    def __init__(self, features, labels=None, batch_size=128, valid_frac=0.1, test_frac=0.1,
                 alter_features=None, alter_labels=None):
        self.batch_size = batch_size
        self.valid_frac = valid_frac
        self.test_frac = test_frac
        self.features = features
        self.labels = labels
        self.alter_features = alter_features
        self.alter_labels = alter_labels

        self.n_rows = {}
        self._setup()
        self._set_partitions()

    def __len__(self):
        return self.n_rows["all"]

    def _setup(self):
        """Execute setup tasks, both input checking and creating derived attributes."""
        # Turn non-Reader data inputs into Readers.
        self.features = get_reader(self.features, labels=False)
        if self.labels is not None:
            self.labels = get_reader(self.labels, labels=True)
        self.n_rows["all"] = len(self.features)

        # Allow None for valid or test fractions.
        if self.valid_frac is None:
            self.valid_frac = 0.
        if self.test_frac is None:
            self.test_frac = 0.

        # Check the inputs.
        if self.labels is not None and len(self.features) != len(self.labels):
            raise ValueError("The features have {} rows, but the labels have {} "
                             "rows.".format(len(self.features), len(self.labels)))

        if self.valid_frac > 1 or self.valid_frac < 0:
            raise ValueError("Select a validation set fraction from [0, 1).")
        if self.test_frac > 1 or self.test_frac < 0:
            raise ValueError("Select a test set fraction from [0, 1).")

        # Figure out where we're starting each section of the data as a fraction of the whole.
        self.n_epochs = {"train": 0, "test": 0, "valid": 0}
        self._start_stop_frac = {"train": (0., 1 - self.valid_frac - self.test_frac),
                                 "valid": (1 - self.valid_frac - self.test_frac, 1 - self.test_frac),
                                 "test": (1 - self.test_frac, None)}
        if self._start_stop_frac["train"][1] <= 0:
            raise ValueError("A validation set of {%:.2} of the data and test set of {%:.2} of "
                             "the data don't leave any training "
                             "data.".format(self.valid_frac, self.test_frac))

        # Translate the start/stop fractions into start/stop rows.
        self.start_stop = {}
        for key, val in self._start_stop_frac.items():
            start_row = self.features.get_start_row(val[0], batch_size=self.batch_size)
            # The `batch_size` input makes sure each section stops at an integer number of batches.
            # Allow the test partition (the last one) to go to the end of the data.
            stop_row = self.features.get_stop_row(start_row, val[1],
                                                  batch_size=(None if key == "test" else
                                                              self.batch_size))
            self.start_stop[key] = (start_row, stop_row)
            self.n_rows[key] = stop_row - start_row

        # Record if there's data partitions we're not using.
        self.using_partition = {"valid": self.valid_frac,
                                "test": self.test_frac,
                                "train": True}

    def _set_partitions(self):
        """Create a `Data` object for training, testing, and validation partitions, and store
        them in this instance."""
        for partition_name in ["train", "test", "valid"]:
            if self.using_partition[partition_name]:
                partition = Data(self.features, self.labels, self.batch_size,
                                 alter_features=self.alter_features,
                                 alter_labels=self.alter_labels,
                                 start=self.start_stop[partition_name][0],
                                 stop=self.start_stop[partition_name][1],
                                 allow_partial_batch=(partition_name != "train"))
            else:
                partition = None
            setattr(self, partition_name, partition)

    def iter_epoch(self, which="train", num_cached=3):
        """Return an iterator which steps through one epoch of the specified partition."""
        if not self.using_partition[which]:
            return
        if which not in self.start_stop:
            raise ValueError("Pick `which` from {}.".format(list(self.start_stop.keys())))

        return getattr(self, which).iter_epoch(num_cached=num_cached)


def get_reader(src, labels=False):
    """Returns a Reader of the appropriate type to iterate over the given source.
    If the source is an HDF5 file, we'll attempt to guess the table name.
    Create the Reader manually if you have an HDF5 file with a non-inferrable table name.
    """
    # If the input is a file, figure out which type.
    if isinstance(src, six.string_types):
        ftype = files.get_file_type(src)
    else:
        ftype = None

    # If the input was the name of a pickle file, read from that pickle.
    # If there's two things inside, then assume it's a tuple of (features, labels).
    # Otherwise assume that the entire thing is what we want.
    if ftype == "pkl":
        data = files.read_pickle(src)
        if len(data) == 2:
            if labels:
                log.debug("Taking the second element of data in {} as our labels.".format(src))
                src = data[1]
            else:
                log.debug("Taking the first element of data in {} as our features.".format(src))
                src = data[0]
        else:
            src = data

    # Turn the input into a Reader, if it isn't already.
    if isinstance(src, (np.ndarray, pd.DataFrame)):
        rdr = ArrayReader(src)
    elif ftype == "hdf":
        # HDF5 file input. Try to infer the proper table name.
        with pd.HDFStore(src, "r") as store:
            keys = [k.strip("/") for k in store.keys()]
            if len(keys) == 1:
                table_name = keys[0]
            else:
                # Assume that a table holds labels if it has one of a standard set of names.
                label_keys = [k for k in keys if k in ["label", "labels", "target", "targets"]]
                if labels:
                    if len(label_keys) == 1:
                        table_name = label_keys[0]
                    else:
                        raise ValueError("I could not infer the name of the table holding labels "
                                         "in {}.".format(src))
                else:
                    if len(keys) - len(label_keys) == 1:
                        table_name = [k for k in keys if k not in label_keys][0]
                    else:
                        raise ValueError("I could not infer the name of the table holding features "
                                         "in {}.".format(src))

        rdr = HDFReader(src, table_name)
    elif isinstance(src, Reader):
        # The input could already be a Reader, in which case we don't need to do anything.
        rdr = src
    else:
        raise TypeError("Could not figure out what to do with data source {}.".format(src))

    return rdr


class Reader(object):
    """
    This is the abstract base class for reading data from various sources.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, data_src):
        self.data_src = data_src

        # Set the following attributes in the subclasses, once we know how to figure
        # this information out from the input data.
        self.n_rows = None
        self.shape = None  # This is the shape of a single row of data.
        self.ndim = None  # The number of dimensions of a single row of data.
        self.dtype = None  # This is the data type of the entire data store, not a single row (which might have mixed dtypes).

    @abc.abstractmethod
    def iter_epoch(self, batch_size, start=0, stop=None, start_on_batch=True, allow_partial=False):
        """Iterate through an opened data source."""

    def __len__(self):
        return self.n_rows

    def get_start_row(self, start, batch_size=None):
        """Figure out which row iteration should start from.
        Translate a fraction-of-the-file input into a row index, and shift
        the start row to lie on the closest previous batch boundary, if
        we're given a `batch_size`.

        **Parameters**

        * `start` <int or float>: Start iterating from here. May be an integer row index
            or a float fraction of the total rows.

        **Optional Parameters**

        * `batch_size` <int|None>: If provided, shift the starting row so that it's the first row
            at or before `start` which is a multiple of `batch_size`.

        **Returns**

        An integer row index.

        **Raises**

        `ValueError` if `start` is bad.
        """
        # Do input checking on the `start`, and convert it to the appropriate row index if needed.
        if isinstance(start, (float, np.floating)):
            # Convert fractional starts to row numbers
            if start >= 1. or start < 0:
                raise ValueError("Fractional start locations must be in [0., 1).")
            start = int(start * self.n_rows)
        if start >= self.n_rows or start < 0:
            # Make sure we're not out of bounds.
            raise ValueError("Can't start at row {} of a {}-row array.".format(start, self.n_rows))
        if batch_size is not None:
            # Often we'll want to start an integer number of batches into the array.
            start = (start // batch_size) * batch_size

        return start

    def get_stop_row(self, start, stop, batch_size=None):
        """Figure out where iteration should end.
        Translate a fraction-of-the-file input into a row index, and shift
        the end row to be such that iteration will cover an integer number of batches, if
        we're given a `batch_size`.

        **Parameters**

        * `start` <int>: Start iterating from here. Used to adjust the end row so that we
            reach an integer number of batches.
        * `stop` <int or float>: Stop iterating here. May be an integer row index or a float
            fraction of the total number of rows.

        **Optional Parameters**

        * `batch_size` <int|None>: If provided, shift the ending row so that it's the first row
            at or before `stop` which is a multiple of `batch_size` away from `start`.

        **Returns**

        An integer row index.

        **Raises**

        `ValueError` if `stop` is bad or `start` is not an integer.
        """
        # If `start` is accidentally a fraction, this won't work.
        if isinstance(start, (float, np.floating)):
            raise ValueError("`start` must be a number of rows, not a fraction.")

        # Do input checking on the `stop`, and convert it to the appropriate row index if needed.
        if stop is None:
            stop = self.n_rows  # Default to taking all of the available rows.
        elif isinstance(stop, (float, np.floating)) or stop == 1:
            if stop > 1. or stop <= 0:
                raise ValueError("Fractional stop locations must be in (0., 1].")
            stop = int(stop * self.n_rows)
        if stop > self.n_rows or stop <= 0:
            raise ValueError("Can't stop at row {} of a {}-row array.".format(stop, self.n_rows))

        # Adjust the `stop` so that it's an integer number of batches from the `start`.
        if batch_size is not None:
            stop = ((stop - start) // batch_size) * batch_size + start

        return stop


class HDFReader(Reader):
    """
    Read from an HDF5 file. We assume that the images are stored in a pandas structure which
    can be cast as an array. The tables should be created appendable so that they have
    all the necessary metadata. Images should be stored as either a Panel or Panel4D.
    For example, you can store a single image in a row of an HDF5 table as

    store = pd.HDFStore(filename)
    for i_row, image in enumerate(all_my_images):
        store.append("labels", labels.iloc[i_row: i_row + 1])
        store.append("images", pd.Panel4D({labels.index[i_row]: image}),
                     axes=["labels", "major_axis", "minor_axis"], complib="zlib", complevel=9)
    store.close()

    Here, "labels" is a Series object containing the labels of all your images, and
    "image" is a 3D array with the color axis first.
    """
    def __init__(self, fname, table=None, color=None, asarray=None):
        """
        * `fname` <str>: File name of an HDF5 file

        * `table` <str|None>: Name of a table in `fname` which contains data. Must be supplied
            if the file has more than one table.

        * `color` <int|None>: Default choice for the `color` input to `iter_epochs`.

        * `asarray` <bool|None>: Cast outputs to arrays? Defaults to True if the rows of
            data have more than 1 dimension, and False for 1D rows.
        """
        super(HDFReader, self).__init__(fname)

        self.color = color
        self.filename = fname
        self.table_name = table

        with pd.HDFStore(self.filename, "r") as data_src:
            if self.table_name is None:
                if len(data_src.keys()) > 1:
                    raise ValueError("The HDF5 file has tables {}: which do you "
                                     "want?".format(data_src.keys()))
                else:
                    self.table_name = data_src.keys()[0].strip("/")

            # Read the first row of data to find the shape.
            # Trim the first element from the shape -- it will be 1, the number of rows we read.
            # Assume that the "rows" of data are designated by the first of the index axes.
            # For a Panel4D, this is "labels". For a Panel, this is "items".
            self._index_name = data_src.get_storer(table).index_axes[0].name
            first_row = data_src.select(table, where="{} == 0".format(self._index_name))
            self.shape = first_row.shape[1:]
            self.ndim = len(self.shape)
            self.dtype = type(first_row)
            if hasattr(first_row, "columns"):
                # Support reading the header from DataFrames.
                self.header = first_row.columns
            else:
                self.header = None

            # Figure out if we should cast the output to arrays.
            if asarray is None:
                asarray = self.ndim > 1
            self.asarray = asarray

            # Figure out how many rows of data are in the table.
            # Pandas stores data of > 2D in row x column format. One dimension of input data
            # will be the "columns", and all the rest will be flattened into rows.
            self._n_cols = data_src.get_storer(table).ncols
            self.n_rows = (data_src.get_storer(table).nrows / (np.prod(self.shape) / self._n_cols))
            if self.n_rows != int(self.n_rows):
                raise ValueError("Table {} appears to have data of shape {}, but I failed to find the "
                                 "correct number of rows.".format(data_src.get_storer(table),
                                                                  self.shape))
            self.n_rows = int(self.n_rows)

        log.debug("Opened file {}. I found {} rows of data with shape "
                  "{}.".format(self.filename, self.n_rows, self.shape))

    def iter_epoch(self, batch_size, start=0, stop=None, start_on_batch=True,
                   allow_partial=False, color=None):
        """
        Iterate through this array, one batch at a time.

        **Parameters**

        * `batch_size` <int>: Number of rows of the array to return at once.

        **Optional Parameters**

        * `start` <int or float|0>: Start at this row. Either an integer row number, or a
            fraction of the total rows. We may start at a slightly different row
            if `start_on_batch` is True.

        * `stop` <int or float|None>: Stop iterating when we reach this many rows. May be given
            as a fraction of the total rows in the array. Will be modified so that we iterate
            through an integer number of batches (unless `allow_partial` is True).
            Default to stopping at the end of the array.

        * `start_on_batch` <bool|True>: If True, modify the `start` row so that we begin at an
            integer number of batches into the array, at or before the requested `start`.

        * `allow_partial` <bool|False>: If False, every returned batch will have `batch_size` rows.
            Iteration will stop at or before the requested `stop` row. If True, the final returned
            batch may have fewer rows, if the requested chunk of data is not an integer number
            of batches.

        * `color` <int|None>: If not None, select this index from the last axis of the shared
            data. For multicolor images, we expect to have shape (rows, columns, colors).
            Will not work if the data are not stored as a Panel or Panel4D.

        **Returns**

        An iterator over portions of the array.
        """
        if color is None:
            color = self.color
        if color is not None and self.dtype not in [pd.Panel, pd.Panel4D]:
            raise ValueError("Cannot select a `color` unless reading image data.")

        start = self.get_start_row(start=start, batch_size=(batch_size if start_on_batch else None))
        stop = self.get_stop_row(start, stop, batch_size=(None if allow_partial else batch_size))

        log.debug("Iterating through HDF5 file {} from row {} to row {} in batches "
                  "of {}.".format(self.filename, start, min([stop, self.n_rows]), batch_size))

        # Set up the iteration.
        array_maker = (lambda x: np.asarray(x, dtype=theano.config.floatX).squeeze()) \
            if self.asarray else (lambda x: x)
        item_size = np.prod(self.shape) / self._n_cols  # A single row of data is this many "rows" in the HDF5Store.
        if color is not None:
            item_size /= self.shape[-1]
        where_stmt = "minor_axis == {color}".format(color=color) if color is not None else None

        # Iterate through the data.
        with pd.HDFStore(self.filename, "r") as data_src:
            for chunk in data_src.select(self.table_name,
                                         start=start * item_size,
                                         stop=stop * item_size,
                                         chunksize=batch_size * item_size,
                                         where=where_stmt):
                yield array_maker(chunk)

        # Code below preserved as a different way of iterating through the table.
        # It's possibly less efficient, and suffers from the flaw of assuming that the
        # index is integers from 0 to n_rows.
        #where_stmt += "({index} >= {start} & {index} < {stop})".format(index=self._index_name,
        #                                                               start="{start}",
        #                                                               stop="{stop}")
        #for i_start in range(start, stop, batch_size):
        #    i_stop = min([i_start + batch_size, stop])
        #    yield array_maker(self.data_src.select(
        #        self.table_name, where=where_stmt.format(start=i_start, stop=i_stop))).squeeze()


class ArrayReader(Reader):
    """
    Read from an array which is entirely in memory.
    """
    def __init__(self, array):
        """Initialize from an input array. The input may also be the file name of a
        pickle which contains a single array.
        """
        if isinstance(array, six.string_types):
            array = files.read_pickle(array)
        super(ArrayReader, self).__init__(np.asarray(array))

        self.n_rows = len(array)
        self.shape = array.shape[1:]
        self.ndim = len(self.shape)
        self.dtype = array.dtype

    def iter_epoch(self, batch_size, start=0, stop=None, start_on_batch=True, allow_partial=False):
        """
        Iterate through this array, one batch at a time.

        **Parameters**

        * `batch_size` <int>: Number of rows of the array to return at once.

        **Optional Parameters**

        * `start` <int or float|0>: Start at this row. Either an integer row number, or a
            fraction of the total rows. We may start at a slightly different row
            if `start_on_batch` is True.

        * `stop` <int or float|None>: Stop iterating when we reach this many rows. May be given
            as a fraction of the total rows in the array. Will be modified so that we iterate
            through an integer number of batches (unless `allow_partial` is True).
            Default to stopping at the end of the array.

        * `start_on_batch` <bool|True>: If True, modify the `start` row so that we begin at an
            integer number of batches into the array, at or before the requested `start`.

        * `allow_partial` <bool|False>: If False, every returned batch will have `batch_size` rows.
            Iteration will stop at or before the requested `stop` row. If True, the final returned
            batch may have fewer rows, if the requested chunk of data is not an integer number
            of batches.

        **Returns**

        An iterator over portions of the array.
        """
        start = self.get_start_row(start=start, batch_size=(batch_size if start_on_batch else None))
        stop = self.get_stop_row(start, stop, batch_size=(None if allow_partial else batch_size))

        log.debug("Iterating through array from row {} to row {} in batches "
                  "of {}.".format(start, min([stop, self.n_rows]), batch_size))

        for i_start in range(start, stop, batch_size):
            yield self.data_src[i_start: min([i_start + batch_size, stop])]
