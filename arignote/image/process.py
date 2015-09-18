"""
This module processes images and makes them suitable to use as neural network inputs.
"""
from __future__ import division


import numpy as np
try:
    from scipy.misc import imresize
    has_scipy = True
except ImportError:
    has_scipy = False

from ..util import misc
from ..util import netlog
log = netlog.setup_logging("image_process", level="INFO")


def rescale_clip(img, new_shape):
    pass


def decorrelate_and_whiten(minibatch):
    """Takes minibatch of images (with batch index first), and zero-means, decorrelates, and
    whitens pixels.
    http://www.slideshare.net/roelofp/python-for-image-understanding-deep-learning-with-convolutional-neural-nets (slide 39)"""
    input_shape = minibatch.shape
    flat_batch = minibatch.reshape((input_shape[0], -1))
    flat_batch = flat_batch - np.mean(flat_batch, axis=0)  # Creating a zero-meaned copy.
    cov = np.dot(flat_batch.T, flat_batch) / flat_batch.shape[0]
    U, S, V = np.linalg.svd(cov)
    batch_rot = np.dot(flat_batch, U)
    batch_white = batch_rot / np.sqrt(S + 1e-5)

    return batch_white.reshape(input_shape)


def find_edges(img, axis, cutoff=0.01):
    """Take an image as input and return the edges of the lighted region.
    Determine "black" by finding regions with less than `cutoff` times the row or column sum.
    """
    # First sum along all dimensions but the one we're interested in.
    axis_index = list(range(img.ndim))
    axis_index.pop(axis)
    axis_sum = np.sum(img, axis=tuple(axis_index))

    # Now find the minima on the high and low side.
    min_low = np.argmin(axis_sum[: len(axis_sum) // 2])
    min_high = np.argmin(axis_sum[len(axis_sum) // 2:]) + len(axis_sum) // 2
    axis_sum = axis_sum[min_low: min_high]

    # Finally, find the edges of the illuminated region.
    edges = np.where(axis_sum > cutoff * axis_sum.max())[0][np.array([0, -1])] + min_low

    return edges


def trim_black(img, cutoff=0.01, axis=(0, 1)):
    """Take an image as input and return an image trimmed so that black regions to the left and
    right and to the top and bottom have been removed. Determine "black" by finding regions
    with less than `cutoff` times the row or column sum.
    """
    axis = misc.as_list(axis)
    for this_axis in axis:
        edges = find_edges(img, axis=this_axis, cutoff=cutoff)
        this_slice = img.ndim * [slice(None)]
        this_slice[this_axis] = slice(*edges)

        img = img[this_slice]
    # column_edges = find_edges(img, axis=1, cutoff=cutoff)
    # width_slice = slice(*column_edges)
    #
    # row_edges = find_edges(img, axis=0, cutoff=cutoff)
    # height_slice = slice(*row_edges)

    return img #[height_slice, width_slice, ...]


def process_images(plk_images, new_scale=32, means=None, stds=None):
    """Prepares images for training. Output training images will have the following transforms:
        - Invert scale (background is 0)
        - Rescale to float in [0, 1]
        - Rescale so that the largest dimension is `new_scale` pixels. Center the smaller
    dimension and zero-pad the edges.

    **Parameters**

    * `plk_images` <list of arrays>: List of arrays, or a 2D array. Each array (or row of a
        2D array) is a greyscale plankton image. The input list will be unchanged.

    **Optional Parameters**

    * `new_scale` <int>: Output images will be (new_scale, new_scale) arrays.

    **Returns**

    A modified list of arrays of the same length as `plk_images`.
    The list has the same length and order as `plk_images`.
    """
    try:
        from skimage import transform
    except ImportError:
        print("This function uses the `transform` function from scikit-image.")
        raise
    output_images = []
    for image in plk_images:

        # Invert image and adjust the range of values
        img = (255 - image) / 255

        # Rescale the image.
        img = transform.rescale(img, new_scale / max(img.shape), order=3, mode="constant", cval=0)
        new_image = np.zeros((new_scale, new_scale))
        if img.shape[0] <= img.shape[1]:
            pdiff = int((new_image.shape[0] - img.shape[0]) / 2)
            new_image[pdiff: pdiff + img.shape[0], :img.shape[1]] = img
        else:
            pdiff = int((new_image.shape[1] - img.shape[1]) / 2)
            new_image[:img.shape[0], pdiff: pdiff + img.shape[1]] = img

        output_images.append(new_image.flatten())

    imgs = np.asarray(output_images)
    # Zero-mean and normalize:
    image_means = np.mean(imgs, axis=0) if means is None else means
    image_std = 1e-7 + np.std(imgs, axis=0) if stds is None else stds  # Regularize to avoid dividing by zero
    output_images = (imgs - image_means) / image_std

    return output_images, image_means, image_std


def create_data_shifts(input_x, input_y, scale):

    shift_x, shift_y = [], []
    for img, label in zip(input_x, input_y):
        img = 255 - img
        imgshifts = shift_image(img, scale)
        for img_ in imgshifts:
            shift_x.append(img_)
            shift_y.append(label)

    shift_x = np.asarray(shift_x)
    shift_y = np.asarray(shift_y)

    image_means = np.mean(shift_x, axis=0)
    image_std = 1e-7 + np.std(shift_x, axis=0)  # Regularize to avoid dividing by zero
    shift_x = (shift_x - image_means) / image_std

    return shift_x, shift_y, image_means, image_std


class ZCA:
    """ Copied from https://gist.github.com/duschendestroyer/5170087
    which itself was based off
    http://ufldl.stanford.edu/wiki/index.php/Implementing_PCA/Whitening
    """
    def __init__(self, regularization=10**-5, copy=False):
        self.regularization = regularization
        self.copy = copy

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mean_ = np.mean(X, axis=0)
        X -= self.mean_
        sigma = np.dot(X.T, X) / X.shape[1]
        U, S, V = np.linalg.svd(sigma)
        tmp = np.dot(U, np.diag(1 / np.sqrt(S + self.regularization)))
        self.components_ = np.dot(tmp, U.T)
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_transformed = X - self.mean_
        X_transformed = np.dot(X_transformed, self.components_.T)
        return X_transformed


def shift_image(img, scale):
    """ Creates 3 images from an input image.  The image
    is scaled so that the min side is equal to the scale.
    After scaling, the image is shifted cropped along
    the max side so that it equals the scale in three
    ways in order to get 3 different "shifts" of the
    image.
    """
    if not has_scipy:
        raise ImportError("This function requires scipy.")

    if img.shape[0] > img.shape[1]:
        img = np.rot90(img)

    # Calculate scaling and shift numbers.
    scale_pct = float(scale) / img.shape[0]
    max_side_size = int(np.round(img.shape[1] * scale_pct))
    diff = max_side_size - scale
    half_diff_floor = int(np.floor(diff/2.0))
    half_diff_ceil = int(np.ceil(diff/2.0))
    img_ = imresize(img, (scale, max_side_size))

    if img_.shape[0] == img_.shape[1]:
        # Add some zero padding so we can get at least a small jitter on this image.
        npad = 3
        padded_img = np.zeros((img_.shape[0], img_.shape[1] + 2 * npad))
        padded_img[:, npad:img_.shape[1] + npad] = img_
        img_ = padded_img

        diff = 2 * npad
        max_side_size = img_.shape[1]
        half_diff_ceil, half_diff_floor = npad, npad

    # Quality check the scaled image.
    assert img_.shape[0] == scale
    assert img_.shape[0] < img_.shape[1]

    # Create shifts.
    img_a = img_[:, diff:max_side_size]
    img_b = img_[:, :max_side_size - diff]
    img_c = img_[:, half_diff_floor:max_side_size - half_diff_ceil]
    flat_images = [x.flatten() for x in [img_a, img_b, img_c]]
    assert all([len(img) == scale * scale for img in flat_images])

    return flat_images
