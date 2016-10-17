import sys
sys.path.append('..')

from collections import OrderedDict
import theano.tensor as T

from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer
from lasagne.layers.merge import ElemwiseMergeLayer
from lasagne.layers.shape import SliceLayer, dimshuffle, reshape
from lasagne.layers.conv import Conv2DLayer, Deconv2DLayer as deconv
from lasagne.nonlinearities import linear, sigmoid, tanh

from pixel_nn_utils import softmax, repeat, get_mask

def get_gated(l_inp, n_units, i, name, latent=None):
    net = OrderedDict()

    if isinstance(latent, InputLayer):
        if len(latent.output_shape) == 2:
            net['lat_proj_{}'.format(i)] = DenseLayer(
                latent, num_units=n_units, name='lat_proj'
            )

            # dimshuffle to match the spatial dimensions
            net['dimshfl_{}'.format(i)] = dimshuffle(
                net.values()[-1], pattern=(0, 1, 'x', 'x'), name='dimshfl'
            )
            lat = repeat([l_inp, net.values()[-1]], name='repeat')
        elif len(latent.output_shape) == 4:
            # y = f(h), using upsampling with deconv layer
            fsize = l_inp.output_shape[-1] - latent.output_shape[-1] + 1
            lat = net['lat_proj_{}'.format(i)] = deconv(
                latent, num_filters=n_units, filter_size=fsize,
                name='lat_proj'
            )
        else:
            raise NotImplementedError
        l_inp = net['lat_merge_{}'.format(i)] = ElemwiseMergeLayer(
                [l_inp, lat], merge_function=T.add, name='lat_merge'
            )

    l_tanh = net['tanh_{}_{}'.format(name, i)] = NonlinearityLayer(
        SliceLayer(l_inp, indices=slice(0, n_units // 2), axis=1),
        nonlinearity=tanh, name='tanh_{}_slice'.format(name)
    )

    l_sigmoid = net['sigmoid_{}_{}'.format(name, i)] = NonlinearityLayer(
        SliceLayer(l_inp, indices=slice(n_units // 2, None), axis=1),
        nonlinearity=sigmoid, name='sigmoid_{}_slice'.format(name)
    )

    net['prod_merge_{}_{}'.format(name, i)] = ElemwiseMergeLayer(
        [l_tanh, l_sigmoid], T.mul, name='prod_merge_{}'.format(name)
    )
    return net.values()[-1], net


def build_pixelcnn_block(incoming_vert, incoming_hor, fsize, i, masked=None,
                         latent=None):
    net = OrderedDict()
    # input (batch_size, n_features, n_rows, n_columns), n_features = p
    assert incoming_vert.output_shape[1] == incoming_hor.output_shape[1]
    nfilts = incoming_hor.output_shape[1] # 2p

    # vertical nxn convolution part, fsize = (n,n)
    if masked:
        # either masked
        net['conv_vert_{}'.format(i)] = Conv2DLayer(
            incoming_vert, num_filters=2*nfilts, filter_size=fsize, pad='same',
            nonlinearity=linear, name='conv_vert'
        ) # 2p
        f_shape = net.values()[-1].W.get_value(borrow=True).shape
        net.values()[-1].W *= get_mask(f_shape, 'A')
    else:
        # or (n//2+1, n) convolution with padding and croppding
        net['conv_vert_{}'.format(i)] = Conv2DLayer(
            incoming_vert, num_filters=2*nfilts, filter_size=(fsize//2+1, fsize),
            pad=(fsize//2+1, fsize//2), nonlinearity=linear, name='conv_vert'
        ) # 2p

        # crop
        net['slice_vert'] = SliceLayer(
            net.values()[-1], indices=slice(0, -fsize//2-1), axis=2,
            name='slice_vert'
        )

    # vertical gated processing
    l_out_vert, gated_vert = get_gated(
        net.values()[-1], 2*nfilts, i, 'vert', latent
    )
    net.update(gated_vert) # p

    # vertical skip connection to horizontal stack
    net['full_conv_vert_{}'.format(i)] = Conv2DLayer(
        l_out_vert, num_filters=2*nfilts, filter_size=1, pad='same',
        nonlinearity=linear, name='full_conv_vert'
    )
    skip_vert2hor = net.values()[-1]

    # horizontal 1xn convolution part, fsize = (1,n)
    if masked:
        net['conv_hor_{}'.format(i)] = Conv2DLayer(
            incoming_hor, num_filters=2*nfilts, filter_size=(1, fsize),
            pad='same', nonlinearity=linear, name='conv_hor'
        ) # 2p
        f_shape = net.values()[-1].W.get_value(borrow=True).shape
        net.values()[-1].W *= get_mask(f_shape, 'A')
    else:
        net['conv_hor_{}'.format(i)] = Conv2DLayer(
            incoming_hor, num_filters=2*nfilts, filter_size=(1, fsize//2+1),
            pad=(0, fsize//2+1), nonlinearity=linear, name='conv_hor'
        ) # 2p

        # crop
        net['slice_hor'] = SliceLayer(
            net.values()[-1], indices=slice(0, -fsize//2-1), axis=3,
            name='slice_hor'
        )

    # merge results of vertical and horizontal convolutions
    net['add_vert2hor_{}'.format(i)] = ElemwiseMergeLayer(
        [skip_vert2hor, net.values()[-1]], T.add, name='add_vert2hor'
    ) # 2p

    # horizontal gated processing
    l_gated_hor, gated_hor = get_gated(
        net.values()[-1], 2*nfilts, i, 'hor', latent
    )
    net.update(gated_hor) # p

    # horizontal full convolution
    net['conv_hor_{}'.format(i)] = Conv2DLayer(
        l_gated_hor, num_filters=nfilts, filter_size=1, pad='same',
        nonlinearity=linear, name='conv_hor'
    )

    # add horizontal skip connection
    net['add_skip2hor_{}'.format(i)] = ElemwiseMergeLayer(
        [net.values()[-1], incoming_hor], T.add, name='add_skip2hor'
    )

    return net, l_out_vert, net.values()[-1] # net, vert output, hor output


def build_pixel_cnn(input_shape, nfilts=384, fsize=5, n_layers=20, masked=True,
                    latent=None):
    net = OrderedDict()
    if input_shape[0] > 1:
        out_dim = 3*256
        out_fn = linear
    else:
        out_dim = 1
        out_fn = sigmoid

    if latent:
        net['latent'] = latent

    net['input'] = InputLayer((None, )+input_shape, name='input')
    net['input_conv'] = Conv2DLayer(
        net.values()[-1], num_filters=nfilts, filter_size=7, pad='same',
        name='input_conv'
    )
    f_shape = net.values()[-1].W.get_value(borrow=True).shape
    net.values()[-1].W *= get_mask(f_shape, 'A')
    l_vert = l_hor = net.values()[-1]

    for i in range(n_layers):
        block, l_vert, l_hor = build_pixelcnn_block(
            l_vert, l_hor, fsize, i, masked, latent=latent
        )
        net.update(block)

    net['pre_relu'] = NonlinearityLayer(net.values()[-1], name='pre_relu')

    net['pre_output'] = Conv2DLayer(
        net.values()[-1], num_filters=nfilts, filter_size=1, name='pre_output'
    ) # contains relu
    f_shape = net.values()[-1].W.get_value(borrow=True).shape
    net.values()[-1].W *= get_mask(f_shape, 'A')

    net['output'] = Conv2DLayer(
        net.values()[-1], num_filters=out_dim, filter_size=1,
        nonlinearity=out_fn, name='output'
    )
    f_shape = net.values()[-1].W.get_value(borrow=True).shape
    net.values()[-1].W *= get_mask(f_shape, 'A')

    if input_shape[0] > 1:
        output_shape = (input_shape[0], 256, 3, input_shape[2], input_shape[3])
        net['output'] = reshape(
            net.values()[-1], shape=output_shape, name='output'
        )

        net['output'] = NonlinearityLayer(
            net.values()[-1], nonlinearity=softmax, name='output'
        )

    return net, net['output']