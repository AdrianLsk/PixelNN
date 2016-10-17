import sys
sys.path.append('..')

from collections import OrderedDict

import numpy as np
import theano.tensor as T

from lasagne.layers import InputLayer, NonlinearityLayer
from lasagne.layers.conv import Conv2DLayer
from lasagne.layers.shape import pad, SliceLayer, reshape
from lasagne.layers.merge import ElemwiseMergeLayer
from lasagne.nonlinearities import sigmoid, linear

from pixel_rnn_layers import SkewLayer, UnSkewLayer, PixelLSTMLayer
from pixel_nn_utils import softmax, get_mask

def build_pixelrnn_block(incoming, i, connected=False, learn_init=True):
    net = OrderedDict()
    num_units = incoming.output_shape[1] // 2

    net['skew_{}'.format(i)] = SkewLayer(incoming, name='skew')
    if connected:
        # igul implementation
        net['rnn_{}'.format(i)] = PixelLSTMLayer(
            net.values()[-1], num_units=num_units, learn_init=learn_init,
            mask_type='B', name='rnn_conn'
        )
        net['bi_rnn_{}'.format(i)] = PixelLSTMLayer(
            net.values()[-1], num_units=num_units, learn_init=learn_init,
            mask_type='B', backwards=True, name='birnn_conn'
        )
    else:
        # original paper says:
        # Given the two output maps, to prevent the layer from seeing future
        # pixels, the right output map is then shifted down by one row and
        # added to the left output map
        skew_l = net.values()[-1]
        rnn_l = net['rnn_{}'.format(i)] = PixelLSTMLayer(
            skew_l, num_units=num_units, precompute_input=True,
            learn_init=learn_init, mask_type='B', name='rnn'
        )
        # W = net.values()[-1].W_in_to_ingate
        # f_shape = np.array(W.get_value(borrow=True).shape)
        # f_shape[1] *= 4
        # W *= get_mask(tuple(f_shape), 'B')

        net['bi_rnn_{}'.format(i)] = PixelLSTMLayer(
            skew_l, num_units=num_units, precompute_input=True,
            learn_init=learn_init, mask_type='B', name='birnn'
        )
        # W = net.values()[-1].W_in_to_ingate
        # f_shape = np.array(W.get_value(borrow=True).shape)
        # f_shape[1] *= 4
        # W *= get_mask(tuple(f_shape), 'B')

        # slice the last row
        net['slice_last_row'] = SliceLayer(
            net.values()[-1], indices=slice(0, -1), axis=2, name='slice_birnn'
        )

        # pad first row with zeros
        net['pad'] = pad(
            net.values()[-1], width=[(1,0)], val=0, batch_ndim=2,
            name='pad_birnn'
        )

        # add together
        net['rnn_out'] = ElemwiseMergeLayer(
            [rnn_l, net.values()[-1]], merge_function=T.add, name='add_rnns'
        )
        
    net['unskew_{}'.format(i)] = UnSkewLayer(net.values()[-1], name='skew')
    
    # 1x1 upsampling by full convolution
    nfilts = incoming.output_shape[1]
    net['full_deconv_{}'.format(i)] = Conv2DLayer(
        net.values()[-1], num_filters=nfilts, filter_size=1, name='full_conv'
    )
        
    # residual skip connection
    net['skip_{}'.format(i)] = ElemwiseMergeLayer(
        [incoming, net.values()[-1]], merge_function=T.add, name='add_rnns')
    
    return net


def build_pixelcnn_block(incoming, i):
    net = OrderedDict()

    nfilts = incoming.output_shape[1] # nfilts = 2h
    net['full_deconv_A_{}'.format(i)] = Conv2DLayer(
        incoming, num_filters=nfilts//2, filter_size=1, name='conv_A'
    )

    net['full_deconv_B_{}'.format(i)] = Conv2DLayer(
        net.values()[-1], num_filters=nfilts//2, filter_size=3, pad='same',
        name='conv_B'
    )
    f_shape = net.values()[-1].W.get_value(borrow=True).shape
    net.values()[-1].W *= get_mask(f_shape, 'B')

    net['full_deconv_C_{}'.format(i)] = Conv2DLayer(
        net.values()[-1], num_filters=nfilts, filter_size=1, name='conv_C'
    )

    # residual skip connection
    net['skip_{}'.format(i)] = ElemwiseMergeLayer(
        [incoming, net.values()[-1]], merge_function=T.add, name='add_convs')

    return net


def build_pixel_nn(dataset='mnist', type='rnn'):
    if dataset == 'mnist':
        input_shape = (1, 28, 28)
        n_layers = 7
        n_units = 16
        top_units = 32
        out_dim = 1
        out_fn = sigmoid
    elif dataset == 'cifar10':
        input_shape = (3, 28, 28)
        n_layers = 12
        n_units = 128
        top_units = 1024
        out_dim = 3*256
        out_fn = linear

    if type == 'cnn':
        n_units = 128
        n_layers = 15

    net = OrderedDict()
    net['input'] = InputLayer((None, )+input_shape, name='input')
    net['input_conv'] = Conv2DLayer(
        net.values()[-1], num_filters=2*n_units, filter_size=7, pad='same',
        name='input_conv'
    )
    f_shape = net.values()[-1].W.get_value(borrow=True).shape
    net.values()[-1].W *= get_mask(f_shape, 'A')

    for i in range(n_layers):
        if type == 'cnn':
            block = build_pixelcnn_block(net.values()[-1], i)
        else:
            block = build_pixelrnn_block(net.values()[-1], i)
        net.update(block)

    net['pre_relu'] = NonlinearityLayer(net.values()[-1], name='pre_relu')

    net['pre_output'] = Conv2DLayer(
        net.values()[-1], num_filters=top_units, filter_size=1, name='pre_output'
    ) # contains relu
    f_shape = net.values()[-1].W.get_value(borrow=True).shape
    net.values()[-1].W *= get_mask(f_shape, 'B')

    net['output'] = Conv2DLayer(
        net.values()[-1], num_filters=out_dim, filter_size=1,
        nonlinearity=out_fn, name='output'
    )
    f_shape = net.values()[-1].W.get_value(borrow=True).shape
    net.values()[-1].W *= get_mask(f_shape, 'B')

    if dataset == 'cifar10':
        output_shape = (input_shape[0], 256, 3, input_shape[2], input_shape[3])
        net['output'] = reshape(
            net.values()[-1], shape=output_shape, name='output'
        )

        net['output'] = NonlinearityLayer(
            net.values()[-1], nonlinearity=softmax, name='output'
        )

    return net, net['output']