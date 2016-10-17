from __future__ import print_function
import os
import sys
sys.path.append('..')

import time

import numpy as np

import theano
import theano.tensor as T
import theano.misc.pkl_utils

from pixel_rnn import build_pixel_nn
from pixel_nn_utils import load_data, save_dump, plot_learning_curves, \
    color_grid_vis, grayscale_grid_vis
from similarity_scores import nnd_score

import lasagne
from lasagne.updates import adam, rmsprop
from lasagne.regularization import l2, regularize_network_params

import argparse

rng = np.random
rng.seed(123)

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-lr','--learning-rate', type=float, default=0.001)
arg_parser.add_argument('-b1','--beta1', type=float, default=0.5)
arg_parser.add_argument('-dlr', '--lr-decay', type=float, default=1.)
arg_parser.add_argument('-l2', '--l2-penalty', type=float, default=None)
arg_parser.add_argument('-bs', '--batch-size', type=int, default=16)
arg_parser.add_argument('-ep','--max-epochs', type=int, default=100)
arg_parser.add_argument('-ndec','--num-decay', type=int, default=0)
arg_parser.add_argument('--show-every', type=int, default=2)
arg_parser.add_argument('--visualize', action='store_true', default=False)
arg_parser.add_argument('--debug', action='store_true', default=False)
arg_parser.add_argument('--binarize', action='store_true', default=False)
arg_parser.add_argument('--small-dataset', action='store_true', default=False)
arg_parser.add_argument('--dataset', type=str, default='MNIST')
arg_parser.add_argument('--model', type=str, default='RNN')
arg_parser.add_argument('--conditional', type=str, default=None)
args = arg_parser.parse_args()


dataset = args.dataset

if dataset == 'MNIST':
    INPUT_SHAPE = (1, 28, 28)
elif dataset == 'CIFAR10':
    INPUT_SHAPE = (3, 32, 32)

LEARNING_RATE = args.learning_rate
LR_DECREASE = args.lr_decay
BATCH_SIZE = args.batch_size
L2 = args.l2_penalty
NUM_EPOCHS = args.max_epochs
NUM_DECAY = args.num_decay
DTYPE = theano.config.floatX

print('Building model ...')
model, output_layer = build_pixel_nn(
    dataset=args.dataset.lower(), type=args.model.lower())
print(map(lambda x: (x.name, x.output_shape), model.values()))

model_pars = lasagne.layers.get_all_params(output_layer, trainable=True)

# TODO: think of using randomstream for input: Z=th_rng.uniform(0,1,shape=..)
# for the input_var

output_train = lasagne.layers.get_output(output_layer)
if args.dataset == 'MNIST':
    output_train = T.clip(output_train, 1e-7, 1-1e-7)
output_val = lasagne.layers.get_output(output_layer)
if args.dataset == 'CIFAR10':
    output_val = output_val.argmax(1)

X = model['input'].input_var
Y = T.ftensor4('y')
inps = [X, Y]

# if args.conditional is not None:
#     inps.append(model['latent'].input_var)
    # inps.insert(1, model['latent'].input_var)

if args.dataset == 'MNIST':
    cost = lasagne.objectives.binary_crossentropy(output_train, Y)
elif args.dataset == 'CIFAR10':
    cost = lasagne.objectives.categorical_crossentropy(
        output_train.reshape(-1, 256), Y.flatten())

cost = lasagne.objectives.aggregate(cost, weights=None, mode='mean')
if args.l2_penalty is not None:
    l2_penalty = regularize_network_params(output_layer, l2) * L2
    cost += l2_penalty

sh_lr = theano.shared(lasagne.utils.np.float32(LEARNING_RATE))
# updates = adam(cost, model_pars, learning_rate=sh_lr)
updates = rmsprop(cost, model_pars, learning_rate=sh_lr)

print('Compiling functions ...')
train_fn = theano.function(inps, cost, updates=updates)
val_fn = theano.function(inps, cost)
generate = theano.function(inps[:-1], output_val)

network_dump = {'model': model,
                'output_layer': output_layer}

print('Loading data ...')
dataset = load_data(False, dataset)

if args.small_dataset:
    ones_idx = dataset['y_train'] == 1
    six_idx = dataset['y_train'] == 6
    X_train = [dataset['X_train'][ones_idx][:ones_idx.sum() // 2],
               dataset['X_train'][six_idx][:six_idx.sum() // 2]]
    X_train = np.vstack(X_train)
    n_train = len(X_train)
    train_idx = np.random.permutation(n_train)
    X_train = X_train[train_idx]
    num_batches_train = n_train // BATCH_SIZE

    ones_idx = dataset['y_valid'] == 1
    six_idx = dataset['y_valid'] == 6
    X_valid = dataset['X_valid'][(ones_idx + six_idx).astype(bool)]
    num_batches_valid = (ones_idx + six_idx).sum() // BATCH_SIZE
else:
    X_train = dataset['X_train']
    X_valid = dataset['X_valid']

    n_train = dataset['num_examples_train']
    num_batches_train = n_train // BATCH_SIZE
    num_batches_valid = dataset['num_examples_valid'] // BATCH_SIZE

mean = X_train.mean()
scale = X_train.std()

tile = 10 # int(np.sqrt(BATCH_SIZE))
tiling = (tile, tile)

def standardize(X, mu=mean, sigma=scale):
    return (X - mu) / (sigma + 1e-15)


def binarize(p):
    return np.random.binomial(1, p).astype(DTYPE)


def generate_samples(gen_fn, batch_size):
    n_chnl, nrow, ncol = INPUT_SHAPE
    images = np.random.rand(batch_size, *INPUT_SHAPE)
    for row in range(nrow):
        for col in range(ncol):
            for rgb in range(n_chnl):
                images[:, rgb, row, col] = \
                    gen_fn(images.astype(DTYPE))[:, rgb, row, col]
    return images

train_costs, val_costs, val_dists = [], [], []
name = 'pixel{model}_{ds}_lr_{lr}_dlr_{dlr}_bs_{bs}_ep_{ep}'.format(
    model=args.model.lower(), ds=args.dataset.lower(), lr=LEARNING_RATE,
    dlr=LR_DECREASE, bs=BATCH_SIZE, ep=NUM_EPOCHS
)

sample_folder = './{}/samples'.format(name)
if not os.path.exists(sample_folder):
    os.makedirs(sample_folder)

output_folder = './{}/results'.format(name)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if args.debug:
    exit()

try:
    for n in range(NUM_EPOCHS+NUM_DECAY):
        print("Epoch {} / {}..".format(n + 1, args.max_epochs))
        train_c = []
        now = time.time()
        for b in range(num_batches_train):
            batch_slice = slice(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)
            pixel_output = (X_train[batch_slice]
                            .reshape(-1, *INPUT_SHAPE))
            pixel_input = standardize(pixel_output)

            if args.binarize:
                pixel_output = binarize(pixel_output)

            train_loss = train_fn(pixel_input, pixel_output)
            train_c.append(train_loss)

            if (b + 1) * BATCH_SIZE < n_train - 1:
                batch_slice = slice(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)
                pixel_output = (X_train[batch_slice]
                                .reshape(-1, *INPUT_SHAPE))
                pixel_input = standardize(pixel_output)

                if args.binarize:
                    pixel_output = binarize(pixel_output)

                train_loss = train_fn(pixel_input, pixel_output)
                train_c.append(train_loss)

        train_loss = float(np.mean(train_c))
        train_costs.append(train_loss)

        val_c, val_samp = [], []
        for b in range(num_batches_valid):
            batch_slice = slice(b * BATCH_SIZE, (b + 1) * BATCH_SIZE)

            if b < 20:
                val_samples = generate_samples(generate, batch_size=20)
                val_samples = val_samples.reshape(-1, np.prod(INPUT_SHAPE))
                val_samp.append(val_samples)

            vc = val_fn(
                standardize(X_valid[batch_slice].reshape(-1, *INPUT_SHAPE)),
                X_valid[batch_slice].reshape(-1, *INPUT_SHAPE)
            )
            val_c.append(vc)

        val_loss = float(np.mean(val_c))
        val_costs.append(val_loss)

        val_samples = np.vstack(val_samp)
        val_d = nnd_score(val_samples, X_valid, metric='euclidean')
        val_dist = float(np.mean(val_d))
        val_dists.append(val_dist)

        if n % args.show_every == 0:
            print("  train cost {:.4f}, val cost {:.4f}, val distance {:.4f},"
                  " took {:.2f} s"
                  .format(train_loss, val_loss, val_dist, time.time()-now))

        # if (n+1) % 10 == 0:
        #     # geometric/exponential annealing
        #     new_lr = sh_lr.get_value() * LR_DECREASE
        #     print("New LR:", new_lr)
        #     sh_lr.set_value(np.float32(new_lr))

        if n > NUM_DECAY and NUM_DECAY > 0:
            # arithmetic/linear annealing
            new_lr = np.float32(sh_lr.get_value() - LEARNING_RATE/NUM_DECAY)
            print("New LR:", new_lr)
            sh_lr.set_value(np.float32(new_lr))

        if (n + 1) % 5 == 0:
            # uncomment if to save the whole network
            save_dump('{}/epoch_{}.pkl'.format(output_folder, n), network_dump)

        if args.visualize:
            X_sample = generate_samples(generate, batch_size=tile**2)
            if args.binarize:
                X_sample = binarize(X_sample)
            save_path = '{}/sample_{}.png'.format(sample_folder, n)
            if args.dataset == 'MNIST':
                grayscale_grid_vis(X_sample[:tile**2], tiling, save_path)
            elif args.dataset == 'CIFAR10':
                color_grid_vis(X_sample[:tile**2], tiling, save_path)

except KeyboardInterrupt:
    pass

save_dump('{}/final_epoch.pkl'.format(output_folder), network_dump)

stats = np.stack([train_costs, val_dists], axis=1)
plot_learning_curves(
    stats, legend=['train cost', 'val dist'], show_immediately=False,
    filename='{}/learn_curves'.format(output_folder)
)