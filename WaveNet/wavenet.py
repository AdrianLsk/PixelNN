import sys
sys.path.append('..')

from collections import OrderedDict
import theano.tensor as T

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.merge import ElemwiseMergeLayer
from lasagne.layers.shape import SliceLayer, pad, dimshuffle, reshape
from lasagne.layers.conv import Conv2DLayer as conv
from lasagne.layers.conv import Deconv2DLayer as deconv
from lasagne.layers.conv import DilatedConv2DLayer as dilate
from lasagne.nonlinearities import linear, sigmoid, tanh, rectify

from pixel_nn_utils import softmax, repeat

def build_wavenet_block(incoming, i, specs, latent=None):
    # input (batch_size, num_units, 1, seq_len)
    net = OrderedDict()

    # dilated causal convolution
    nfilts = incoming.output_shape[1]
    fsize, dil = specs # nfilts = 2p, fsize=(1,fs), dil=(1,d)
    net['dil_causal_conv_{}'.format(i)] = pad(
        dilate(incoming, num_filters=nfilts, filter_size=(1, fsize),
               dilation=(1, dil)), width=[(dil, 0)], val=0, batch_ndim=3,
        name='dil_causal_conv')
    l_inp = net.values()[-1]

    if isinstance(latent, InputLayer):
        if len(latent.output_shape) == 2:
            net['lat_proj_{}'.format(i)] = DenseLayer(
                latent, num_units=nfilts, name='lat_proj'
            )

            # dimshuffle to match the spatial dimensions
            net['dimshfl_{}'.format(i)] = dimshuffle(
                net.values()[-1], pattern=(0, 1, 'x', 'x'), name='dimshfl'
            )
            lat = repeat([l_inp, net.values()[-1]], name='repeat')
        elif len(latent.output_shape) == 4:
            # y = f(h), using upsampling with deconv layer
            # input_l = lasagne.layers.helper.get_all_layers(l_inp)[0]
            fsize = l_inp.output_shape[-1] - latent.output_shape[-1] + 1
            lat = net['lat_proj_{}'.format(i)] = deconv(
                latent, num_filters=nfilts, filter_size=(1, fsize),
                name='lat_proj'
            )
        else:
            raise NotImplementedError
        # print(l_inp.output_shape, lat.output_shape)
        l_inp = net['lat_merge_{}'.format(i)] = ElemwiseMergeLayer(
                [l_inp, lat], merge_function=T.add, name='lat_merge'
            )

    # print(l_inp.output_shape)
    # tanh gate
    tanh_sl = net['tanh_slice_{}'.format(i)] = NonlinearityLayer(
        SliceLayer(l_inp, indices=slice(0, nfilts//2), axis=1),
        nonlinearity=tanh, name='tanh_slice'
    )

    # sigmoid gate
    sigmoid_sl = net['sigmoid_slice_{}'.format(i)] = NonlinearityLayer(
        SliceLayer(l_inp, indices=slice(nfilts//2, None), axis=1),
        nonlinearity=sigmoid, name='sigmoid_slice'
    )

    # elementwise merging by pro
    net['prod_merge_{}'.format(i)] = ElemwiseMergeLayer(
        [tanh_sl, sigmoid_sl], T.mul, name='prod_merge'
    )

    # skip connection
    skip = net['full_conv_{}'.format(i)] = conv(
        net.values()[-1], num_filters=nfilts, filter_size=1,
        nonlinearity=linear, name='full_conv'
    )

    # elementwise mergig by addition
    net['res_out_{}'.format(i)] = ElemwiseMergeLayer(
        [incoming, skip], T.add, name='res_out'
    )

    # print(net.values()[-1].input_shapes)
    return skip, net


def build_wavenet(n_channels, seq_length, specs, out_dim=256, out_fn=softmax,
                  latent=None):
    net = OrderedDict()

    if latent:
        net['latent'] = latent

    net['input'] = InputLayer((None, seq_length, n_channels), name='input')
    input_shape = net['input'].input_var.shape
    net['input_dimshfl'] = dimshuffle(
        net.values()[-1], pattern=(0, 2, 'x', 1), name='input_dimshfl'
    )

    nfilts, fsize = specs.pop(0)
    net['causal_conv_0'] = pad(
        conv(net.values()[-1], num_filters=nfilts, filter_size=(1, fsize)),
        width=[(fsize-1, 0)], val=0, batch_ndim=3, name='causal_conv'
    )

    skips = []
    for i, spec in enumerate(specs):
        l_inp = net.values()[-1]
        skip, wavenet_block = build_wavenet_block(
            l_inp, i+1, specs.pop(0), latent=latent
        )
        skips.append(skip)
        net.update(wavenet_block)

    net['skip_merge'] = NonlinearityLayer(
        ElemwiseMergeLayer([l_inp] + skips, T.add), nonlinearity=rectify,
        name='skip_merge'
    )

    net['pre_out'] = conv(
        net.values()[-1], num_filters=nfilts, filter_size=1, nonlinearity=rectify,
        name='pre_out'
    )

    # num_filters = ouput_dim
    net['output'] = conv(
        net.values()[-1], num_filters=out_dim, filter_size=1,
        nonlinearity=out_fn, name='output'
    )

    net['output_dimshfl'] = dimshuffle(
        net.values()[-1], pattern=(0, 3, 1, 2), name='output_dimshfl'
    )

    output_shape = (input_shape[0], input_shape[1], out_dim)
    net['output_reshape'] = reshape(
        net.values()[-1], shape=output_shape, name='output_reshape'
    )

    return net, net.values()[-1]