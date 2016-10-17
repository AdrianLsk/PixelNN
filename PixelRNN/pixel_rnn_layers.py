import numpy as np
import theano
import theano.tensor as T

from lasagne.layers import Layer, MergeLayer, Gate
from lasagne import nonlinearities
from lasagne import init
from lasagne.utils import unroll_scan
from lasagne.theano_extensions.padding import pad
from lasagne.theano_extensions.conv import conv1d_mc1


class SkewLayer(Layer):
    def __init__(self, incoming, **kwargs):
        super(SkewLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[3] = 2*output_shape[3]-1
        return output_shape
    
    def get_output_for(self, input):
        inp_shp = input.shape
        output = T.zeros((inp_shp[0], inp_shp[1], inp_shp[2],
                          inp_shp[2]+inp_shp[3]-1), input.dtype)
        # shp = input.shape
        # for i in range(self.input_shape[2]):
        #     output = T.set_subtensor(
        #         output[:, :, i, i:i+shp[3]], input[:, :, i])

        shp = self.input_shape
        idx = np.indices([shp[2], shp[3]])
        idx[1] = np.arange(shp[2])[:, None] + np.arange(shp[3])[None]
        output = T.set_subtensor(output[:, :, idx[0], idx[1]], input)
        return output


class UnSkewLayer(Layer):
    def __init__(self, incoming, **kwargs):
        super(UnSkewLayer, self).__init__(incoming, **kwargs)

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)
        output_shape[3] = (output_shape[3]+1) // 2
        return output_shape
    
    def get_output_for(self, input):
        shp = self.input_shape
        # shp = input.shape
        # output = T.stack([input[:, :, i, i:i+(shp[3]+1)//2]
        #                   for i in range(self.input_shape[2])], axis=2)

        idx = np.indices([shp[2], (shp[3]+1)//2])
        idx[1] = np.arange(shp[2])[:, None] + np.arange((shp[3]+1)//2)[None]
        return input[:, :, idx[0], idx[1]]


class PixelLSTMLayer(MergeLayer):
    def __init__(self, incoming, num_units,
                 ingate=Gate(),
                 forgetgate=Gate(),
                 cell=Gate(W_cell=None, nonlinearity=nonlinearities.tanh),
                 outgate=Gate(),
                 nonlinearity=nonlinearities.tanh,
                 cell_init=init.Constant(0.),
                 hid_init=init.Constant(0.),
                 backwards=False,
                 learn_init=False,
                 peepholes=False,
                 gradient_steps=-1,
                 grad_clipping=0,
                 unroll_scan=False,
                 precompute_input=True,
                 mask_input=None,
                 mask_type=None,
                 # only_return_final=False,
                 **kwargs):

        # This layer inherits from a MergeLayer, because it can have four
        # inputs - the layer input, the mask, the initial hidden state and the
        # inital cell state. We will just provide the layer input as incomings,
        # unless a mask input, inital hidden state or initial cell state was
        # provided.
        only_return_final=False
        incomings = [incoming]
        self.mask_incoming_index = -1
        self.hid_init_incoming_index = -1
        self.cell_init_incoming_index = -1
        if mask_input is not None:
            incomings.append(mask_input)
            self.mask_incoming_index = len(incomings)-1
        if isinstance(hid_init, Layer):
            incomings.append(hid_init)
            self.hid_init_incoming_index = len(incomings)-1
        if isinstance(cell_init, Layer):
            incomings.append(cell_init)
            self.cell_init_incoming_index = len(incomings)-1

        # Initialize parent layer
        super(PixelLSTMLayer, self).__init__(incomings, **kwargs)

        # If the provided nonlinearity is None, make it linear
        if nonlinearity is None:
            self.nonlinearity = nonlinearities.identity
        else:
            self.nonlinearity = nonlinearity

        self.learn_init = learn_init
        self.num_units = num_units
        self.backwards = backwards
        self.peepholes = peepholes
        self.gradient_steps = gradient_steps
        self.grad_clipping = grad_clipping
        self.unroll_scan = unroll_scan
        self.precompute_input = precompute_input
        self.only_return_final = only_return_final
        self.mask_type = mask_type

        if unroll_scan and gradient_steps != -1:
            raise ValueError(
                "Gradient steps must be -1 when unroll_scan is true.")

        # Retrieve the dimensionality of the incoming layer
        input_shape = self.input_shapes[0]

        if unroll_scan and input_shape[1] is None:
            raise ValueError("Input sequence length cannot be specified as "
                             "None when unroll_scan is True")

        # num_inputs = np.prod(input_shape[2:])
        num_inputs = input_shape[1]

        def add_gate_params(gate, gate_name):
            """ Convenience function for adding layer parameters from a Gate
            instance. """
            # W_shape = (num_inputs, num_units)
            W_shape_2D = (num_units, num_inputs, 1, 1)
            W_shape_1D = (num_units, num_units, 2)
            # return (self.add_param(gate.W_in, (num_inputs, num_units),
            return (self.add_param(gate.W_in, W_shape_2D,
                                   name="W_in_to_{}".format(gate_name)),
                    # self.add_param(gate.W_hid, (num_units, num_units),
                    self.add_param(gate.W_hid, W_shape_1D,
                                   name="W_hid_to_{}".format(gate_name)),
                    self.add_param(gate.b, (num_units,),
                                   name="b_{}".format(gate_name),
                                   regularizable=False),
                    gate.nonlinearity)

        # Add in parameters from the supplied Gate instances
        (self.W_in_to_ingate, self.W_hid_to_ingate, self.b_ingate,
         self.nonlinearity_ingate) = add_gate_params(ingate, 'ingate')

        (self.W_in_to_forgetgate, self.W_hid_to_forgetgate, self.b_forgetgate,
         self.nonlinearity_forgetgate) = add_gate_params(forgetgate,
                                                         'forgetgate')

        (self.W_in_to_cell, self.W_hid_to_cell, self.b_cell,
         self.nonlinearity_cell) = add_gate_params(cell, 'cell')

        (self.W_in_to_outgate, self.W_hid_to_outgate, self.b_outgate,
         self.nonlinearity_outgate) = add_gate_params(outgate, 'outgate')

        # If peephole (cell to gate) connections were enabled, initialize
        # peephole connections.  These are elementwise products with the cell
        # state, so they are represented as vectors.
        if self.peepholes:
            self.W_cell_to_ingate = self.add_param(
                ingate.W_cell, (num_units, ), name="W_cell_to_ingate")

            self.W_cell_to_forgetgate = self.add_param(
                forgetgate.W_cell, (num_units, ), name="W_cell_to_forgetgate")

            self.W_cell_to_outgate = self.add_param(
                outgate.W_cell, (num_units, ), name="W_cell_to_outgate")

        # Setup initial values for the cell and the hidden units
        if isinstance(cell_init, Layer):
            self.cell_init = cell_init
        else:
            self.cell_init = self.add_param(
                cell_init, (1, self.num_units, self.input_shapes[0][-1]),
                name="cell_init", trainable=learn_init, regularizable=False)

        if isinstance(hid_init, Layer):
            self.hid_init = hid_init
        else:
            # (batch_size, num_feats, seq_len)
            # (batch_size, num_input_channels, input_length)
            self.hid_init = self.add_param(
                hid_init, (1, self.num_units, self.input_shapes[0][-1]),
                name="hid_init", trainable=learn_init, regularizable=False)

    def get_output_shape_for(self, input_shapes):
        # The shape of the input to this layer will be the first element
        # of input_shapes, whether or not a mask input is being used.
        input_shape = input_shapes[0]
        # When only_return_final is true, the second (sequence step) dimension
        # will be flattened
        output_shape = list(input_shape)
        output_shape[1] = self.num_units
        return output_shape

    def get_mask(self, filter_shape, mask_type):
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

    def get_output_for(self, inputs, **kwargs):
        # Retrieve the layer input
        input = inputs[0]
        # Retrieve the mask when it is supplied
        mask = None
        hid_init = None
        cell_init = None
        if self.mask_incoming_index > 0:
            mask = inputs[self.mask_incoming_index]
        if self.hid_init_incoming_index > 0:
            hid_init = inputs[self.hid_init_incoming_index]
        if self.cell_init_incoming_index > 0:
            cell_init = inputs[self.cell_init_incoming_index]

        # Stack input weight matrices into a (num_inputs, 4*num_units)
        # matrix, which speeds up computation
        W_in_stacked = T.concatenate(
            [self.W_in_to_ingate, self.W_in_to_forgetgate,
             self.W_in_to_cell, self.W_in_to_outgate], axis=0)

        filter_shape = list(self.W_in_to_ingate.get_value(borrow=True).shape)
        filter_shape[0] *= 4
        W_in_stacked *= self.get_mask(tuple(filter_shape), self.mask_type)

        # Same for hidden weight matrices
        W_hid_stacked = T.concatenate(
            [self.W_hid_to_ingate, self.W_hid_to_forgetgate,
             self.W_hid_to_cell, self.W_hid_to_outgate], axis=0)

        # Stack biases into a (4*num_units) vector
        b_stacked = T.concatenate(
            [self.b_ingate, self.b_forgetgate,
             self.b_cell, self.b_outgate], axis=0).dimshuffle('x', 0, 'x', 'x')

        # border_mode = (self.num_units // 2, self.num_units // 2)

        if self.precompute_input:
            # Because the input is given for all time steps, we can
            # precompute_input the inputs dot weight matrices before scanning.
            # W_in_stacked is (n_features, 4*num_units). input is then
            # (batch_size, 4*num_units, num_rows, num_columns)
            # (n_batch, 4*num_units, height, width).
            input = T.nnet.conv2d(input, W_in_stacked,  None, None,
                                  subsample=(1,1), border_mode='half',
                                  filter_flip=False) + b_stacked

        # Because scan iterates over the first dimension we dimshuffle to
        # (n_time_steps, n_batch, n_features)
        # (height, n_batch, 4*num_units, width)
        # (n_batch, num_units, width)
        input = input.dimshuffle(2, 0, 1, 3)
        seq_len, num_batch = input.shape[:2]

        # When theano.scan calls step, input_n will be (n_batch, 4*num_units).
        # We define a slicing function that extract the input to each LSTM gate
        def slice_w(x, n):
            return x[:, n*self.num_units:(n+1)*self.num_units]

        # Create single recurrent computation step function
        # input_n is the n'th vector of the input
        def step(input_n, cell_previous, hid_previous, *args):
            if not self.precompute_input:
                input_n = T.nnet.conv2d(input_n, W_in_stacked, None, None,
                                        subsample=(1,1), border_mode='half',
                                        filter_flip=False) + b_stacked

            # Calculate gates pre-activations and slice
            hid_previous = pad(hid_previous, [(1, 0)], 0, 2)

            gates = input_n + conv1d_mc1(hid_previous, W_hid_stacked,
                                         None, None, subsample=(1,),
                                         border_mode='valid',
                                         filter_flip=False)

            # Clip gradients
            if self.grad_clipping:
                gates = theano.gradient.grad_clip(
                    gates, -self.grad_clipping, self.grad_clipping)

            # Extract the pre-activation gate values
            ingate = slice_w(gates, 0)
            forgetgate = slice_w(gates, 1)
            cell_input = slice_w(gates, 2)
            outgate = slice_w(gates, 3)

            if self.peepholes:
                # Compute peephole connections
                ingate += cell_previous*self.W_cell_to_ingate
                forgetgate += cell_previous*self.W_cell_to_forgetgate

            # Apply nonlinearities
            ingate = self.nonlinearity_ingate(ingate)
            forgetgate = self.nonlinearity_forgetgate(forgetgate)
            cell_input = self.nonlinearity_cell(cell_input)

            # Compute new cell value
            cell = forgetgate*cell_previous + ingate*cell_input

            if self.peepholes:
                outgate += cell*self.W_cell_to_outgate
            outgate = self.nonlinearity_outgate(outgate)

            # Compute new hidden unit activation
            hid = outgate*self.nonlinearity(cell)
            return [cell, hid]

        def step_masked(input_n, mask_n, cell_previous, hid_previous, *args):
            cell, hid = step(input_n, cell_previous, hid_previous, *args)

            # Skip over any input with mask 0 by copying the previous
            # hidden state; proceed normally for any input with mask 1.
            cell = T.switch(mask_n, cell, cell_previous)
            hid = T.switch(mask_n, hid, hid_previous)

            return [cell, hid]

        if mask is not None:
            # mask is given as (batch_size, seq_len). Because scan iterates
            # over first dimension, we dimshuffle to (seq_len, batch_size) and
            # add a broadcastable dimension
            mask = mask.dimshuffle(1, 0, 'x')
            sequences = [input, mask]
            step_fun = step_masked
        else:
            sequences = input
            step_fun = step

        ones = T.ones((num_batch, 1))
        if not isinstance(self.cell_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            # cell_init = T.dot(ones, self.cell_init)
            cell_init = T.tensordot(ones, T.unbroadcast(self.cell_init, 0),
                                    axes=[1,0])

        if not isinstance(self.hid_init, Layer):
            # Dot against a 1s vector to repeat to shape (num_batch, num_units)
            # hid_init = T.dot(ones, self.hid_init)
            hid_init = T.tensordot(ones, T.unbroadcast(self.hid_init, 0),
                                   axes=[1,0])

        # print(self.cell_init.ndim, self.cell_init.broadcastable)
        # print(cell_init.ndim, cell_init.broadcastable)
        # print(self.hid_init.ndim, self.hid_init.broadcastable)
        # print(hid_init.ndim, hid_init.broadcastable)

        # print(self.cell_init.get_value(True).shape)
        # print(self.hid_init.get_value(True).shape)

        # The hidden-to-hidden weight matrix is always used in step
        non_seqs = [W_hid_stacked]
        # The "peephole" weight matrices are only used when self.peepholes=True
        if self.peepholes:
            non_seqs += [self.W_cell_to_ingate,
                         self.W_cell_to_forgetgate,
                         self.W_cell_to_outgate]

        # When we aren't precomputing the input outside of scan, we need to
        # provide the input weights and biases to the step function
        if not self.precompute_input:
            non_seqs += [W_in_stacked, b_stacked]

        if self.unroll_scan:
            # Retrieve the dimensionality of the incoming layer
            input_shape = self.input_shapes[0]
            # Explicitly unroll the recurrence instead of using scan
            cell_out, hid_out = unroll_scan(
                fn=step_fun,
                sequences=sequences,
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                non_sequences=non_seqs,
                n_steps=input_shape[1])
        else:
            # Scan op iterates over first dimension of input and repeatedly
            # applies the step function
            cell_out, hid_out = theano.scan(
                fn=step_fun,
                sequences=sequences, # input
                outputs_info=[cell_init, hid_init],
                go_backwards=self.backwards,
                truncate_gradient=self.gradient_steps,
                non_sequences=non_seqs,
                strict=True)[0]

        # When it is requested that we only return the final sequence step,
        # we need to slice it out immediately after scan is applied
        if self.only_return_final:
            hid_out = hid_out[-1]
        else:
            # dimshuffle back to (n_batch, n_time_steps, n_features))
            # hid_out = hid_out.dimshuffle(1, 0, 2)
            hid_out = hid_out.dimshuffle(1, 2, 0, 3)

            # if scan is backward reverse the output
            if self.backwards:
                hid_out = hid_out[:, :, ::-1]

        return hid_out