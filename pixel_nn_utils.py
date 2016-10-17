from PIL import Image
import os
import gzip
import cPickle

import lasagne

import theano
import theano.tensor as T
import theano.misc.pkl_utils

import numpy as np


DATA_PATH = os.path.join(os.path.expanduser('~'), 'data')


class RepeatLayer(lasagne.layers.MergeLayer):
    def __init__(self, incomings, **kwargs):
        super(RepeatLayer, self).__init__(incomings, **kwargs)

    def get_output_shape_for(self, input_shapes):
        x_shp, y_shp = input_shapes
        shp_out = list(x_shp)
        shp_out[1] = y_shp[1]
        return shp_out

    def get_output_for(self, inputs, **kwargs):
        x, y = inputs
        return y*T.ones((x.shape[0], y.shape[1], x.shape[2], x.shape[3]))

repeat = RepeatLayer


def get_mask(filter_shape=(1,1,6,6), mask_type='A'):
    fsize_new, fsize_old, rsize, csize = filter_shape
    mask = np.zeros((fsize_new, fsize_old, rsize*csize))
    limit = csize * (rsize // 2) + csize // 2
    mask[:, :, :limit+1] = 1

    pattern = np.zeros((3,3))
    idx = np.arange(3)
    if (mask_type == 'B'):
        idx = idx[:, None] >= idx[None]
    else:
        idx = idx[:, None] > idx[None]

    pattern[idx] = 1
    pattern = pattern[:, np.arange(fsize_old) % 3][np.arange(fsize_new) % 3]

    mask[:, :, limit] *= pattern[:, :]
    mask = mask.reshape(filter_shape)
    return mask.astype(theano.config.floatX)


def softmax(vec, axis=1):
    """
     The ND implementation of softmax nonlinearity applied over a specified
     axis, which is by default the second dimension.
    """
    xdev = vec - vec.max(axis, keepdims=True)
    rval = T.exp(xdev)/(T.exp(xdev).sum(axis, keepdims=True))
    return rval


def load_dump(file):
    if isinstance(file, str):
        fo = open(file, 'rb')
    else:
        fo = file
    obj = cPickle.load(fo)
    fo.close()
    return obj


def save_dump(filename, to_dump, method='cPickle'):
    if method == 'cPickle':
        with open(filename, 'wb') as f:
            cPickle.dump(to_dump, f, protocol=cPickle.HIGHEST_PROTOCOL)
    else:
        theano.misc.pkl_utils.dump(to_dump, filename)


def load_data(shared_var=True, data='MNIST', data_path=DATA_PATH):
    """Get data with labels, split into training, validation and test set."""
    if data == 'MNIST':
        with gzip.open(os.path.join(data_path, 'mnist.pkl.gz'), 'rb') as f:
            data = load_dump(f)
        X_train, y_train = data[0]
        X_valid, y_valid = data[1]
        X_test, y_test = data[2]
    elif data == 'CIFAR10':
        from glob import glob

        batch_files = sorted(
            [x for x in glob(os.path.join(DATA_PATH, 'cifar-10-batches-py/*'))
             if '_batch' in x]
            )

        data = map(load_dump, batch_files)

        X_train = np.vstack(map(lambda x: x['data'],
                                data[:-1])) # .reshape(-1, 3, 32, 32)
        y_train = np.hstack(map(lambda x: x['labels'],
                                data[:-1]))

        shfl_idx = np.random.choice(len(X_train), len(X_train), replace=False)
        X_valid = X_train[shfl_idx][-7500:]
        y_valid = y_train[shfl_idx][-7500:]

        X_train = X_train[shfl_idx][:-7500]
        y_train = y_train[shfl_idx][:-7500]

        X_test = data[-1]['data'] # .reshape(-1, 3, 32, 32)
        y_test = np.array(data[-1]['labels'])

    if shared_var:
        return dict(
            X_train=theano.shared(lasagne.utils.np.float32(X_train)),
            y_train=T.cast(theano.shared(y_train), 'int32'),
            X_valid=theano.shared(lasagne.utils.np.float32(X_valid)),
            y_valid=T.cast(theano.shared(y_valid), 'int32'),
            X_test=theano.shared(lasagne.utils.np.float32(X_test)),
            y_test=T.cast(theano.shared(y_test), 'int32'),
            num_examples_train=X_train.shape[0],
            num_examples_valid=X_valid.shape[0],
            num_examples_test=X_test.shape[0],
            input_dm=X_train.shape[1],
            output_dim=10,
        )
    else:
        return dict(
            X_train=np.float32(X_train),
            y_train=np.int32(y_train),
            X_valid=np.float32(X_valid),
            y_valid=np.int32(y_valid),
            X_test=np.float32(X_test),
            y_test=np.int32(y_test),
            num_examples_train=X_train.shape[0],
            num_examples_valid=X_valid.shape[0],
            num_examples_test=X_test.shape[0],
            input_dm=X_train.shape[1],
            output_dim=10,
        )


def color_grid_vis(X, (nh, nw), save_path=None):
    h, w = X[0].shape[-2:]
    img = np.zeros((h*nh, w*nw, 3))
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w, :] = x
    im_new = Image.fromarray(img.astype('uint8'))
    if save_path is not None:
        im_new.save(save_path)
    else:
        return im_new


def grayscale_grid_vis(X, (nh, nw), save_path=None):
    h, w = X[0].shape[-2:]
    img = np.zeros((h*nh, w*nw))
    for n, x in enumerate(X):
        j = n/nw
        i = n%nw
        img[j*h:j*h+h, i*w:i*w+w] = x
    im_new = Image.fromarray(img.astype('uint8'))
    if save_path is not None:
        im_new.save(save_path)
    else:
        return im_new


def plot_learning_curves(costs, legend, filename=None, figsize=[10, 7],
                         show_immediately=False):
    """
    Save the plot into a png file.

    Parameters
    ----------
    costs : list of np.ndarrays of lists of the same length
        Training and validation costs.
    filename: str
        Name of the file to save the plot.

    Returns
    -------
        None
    """
    import matplotlib.pyplot as plt
    data = zip(*costs)
    plt.figure(figsize=figsize)
    for stats in data:
        plt.plot(stats)
    plt.legend(legend, fontsize=20)
    if filename:
        plt.savefig(filename+'.png')
    if show_immediately:
        plt.show()


def softmax(vec, axis=1):
    """
     The ND implementation of softmax nonlinearity applied over a specified
     axis, which is by default the second dimension.
    """
    xdev = vec - vec.max(axis, keepdims=True)
    rval = T.exp(xdev)/(T.exp(xdev).sum(axis, keepdims=True))
    return rval