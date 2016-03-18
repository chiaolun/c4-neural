#!/usr/bin/env python

import numpy as np
import time
import theano
import theano.tensor as T
import lasagne

# Code copied from lasagne:
# https://github.com/Lasagne/Lasagne/blob/5fcc4aa80fc9a299fbe444a2fa490333a8af6142/LICENSE

ncols = 7
nrows = 6


def parse_game(line0):
    line0 = [int(char0) for char0 in line0.strip()]
    winner = line0[0]
    moves = line0[1:]
    return winner, moves


def moves_to_state(moves, state0=np.zeros((2, nrows, ncols), dtype="int8")):
    state0 = state0.copy()
    top_cells = state0.max(axis=0).argmin(axis=0)
    side = top_cells.sum() % 2
    for move0 in moves:
        state0[side, top_cells[move0], move0] = 1
        top_cells[move0] += 1
        side = 1 - side
    return state0


def sars((winner0, moves0)):
    no_state = np.zeros((2, nrows, ncols))
    sample_move = np.random.randint(0, len(moves0))
    col0 = moves0[sample_move]
    state0 = moves_to_state(moves0[:sample_move])
    side0 = state0.sum() % 2
    flip0 = 1 - 2 * side0
    moves0 = moves0[sample_move:]
    if len(moves0) == 2:
        return state0[::flip0], col0, 1, no_state
    elif len(moves0) == 1:
        return state0[::flip0], col0, -1, no_state
    else:
        state1 = moves_to_state(moves0[:2], state0)
        return state0[::flip0], col0, 0, state1[::flip0]


def gen_batch(games, n):
    sample_games = [games[i] for i in np.random.randint(len(games), size=n)]
    return map(sars, sample_games)


def iterate_minibatches(*arrays, **options):
    batchsize = options.pop("batchsize")
    shuffle = options.pop("shuffle", False)
    if shuffle:
        indices = np.arange(len(arrays[0]))
        np.random.shuffle(indices)
    for start_idx in range(0, len(arrays[0]) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield tuple(x[excerpt] for x in arrays)


def get_network():
    network = lasagne.layers.InputLayer(
        shape=(None, 2, nrows, ncols)
    )
    network = lasagne.layers.Conv2DLayer(
        network, 32, (4, 4),
    )
    network = lasagne.layers.DenseLayer(
        network, num_units=8,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    network = lasagne.layers.DenseLayer(
        network, num_units=ncols,
        nonlinearity=None,
    )
    return network


def compile_Q(network):
    state = T.tensor4('state')
    Q = lasagne.layers.get_output(network, inputs=state)
    Q_fn = theano.function([state], Q)
    return Q_fn


def compile_trainer(network):
    # Prepare Theano variables for inputs and targets
    alpha = T.scalar("alpha")
    state0 = T.tensor4('state0')
    action = T.bvector('action')
    reward = T.vector('reward')
    state1 = T.tensor4('state1')

    # Create a loss expression for training, i.e., a scalar objective
    # we want to minimize
    Q0 = lasagne.layers.get_output(network, inputs=state0)
    Q1 = lasagne.layers.get_output(network, inputs=state1)

    # Q0[action] == reward + alpha * max(Q1) + error
    terminal = T.eq(state1.sum(axis=(1, 2, 3)), 0)
    error_vec = (
        (
            Q0 *
            T.extra_ops.to_one_hot(action, ncols)
        ).sum(axis=1) -
        reward -
        T.switch(
            terminal,
            0.,
            alpha * Q1.max(axis=1)
        )
    )
    error = (error_vec**2).mean()

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(error, params)

    # Compile a function performing a training step on a mini-batch
    # (by giving the updates dictionary) and returning the
    # corresponding training loss:
    train_fn = theano.function(
        [state0, action, reward, state1, alpha],
        error,
        updates=updates,
        on_unused_input='warn')

    return train_fn


games = [parse_game(line0) for line0 in file("RvR.txt").readlines()]

network = get_network()
Q_fn = compile_Q(network)
train_fn = compile_trainer(network)

# Finally, launch the training loop.
print("Starting training...")

num_epochs = 100
# We iterate over epochs:
for epoch in range(num_epochs):
    start_time = time.time()
    state0s, actions, rewards, state1s = zip(*gen_batch(games, 100000))
    state0s = np.array(state0s)
    actions = np.array(actions, dtype="int8")
    rewards = np.array(rewards, dtype="float32")
    state1s = np.array(state1s)

    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    for batch in iterate_minibatches(
            state0s, actions, rewards, state1s,
            batchsize=500, shuffle=True,
    ):
        train_err += train_fn(*(batch + (0.9,)))
        train_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
