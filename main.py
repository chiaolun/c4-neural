#!/usr/bin/env python

import numpy as np
import time
import theano
import theano.tensor as T
import lasagne

ncols = 7
nrows = 6


def parse_game(line0):
    line0 = [int(char0) for char0 in line0.strip()]
    winner = line0[0]
    moves = line0[1:]
    return winner, moves


def moves_to_state(moves):
    state0 = np.zeros((2, nrows, ncols), dtype="int32")
    top_cells = [0] * ncols
    for i, move0 in enumerate(moves):
        state0[i % 2, top_cells[move0], move0] = 1
        top_cells[move0] += 1
    return state0


def moves_to_states(moves, n=1):
    state0 = np.zeros((2, nrows, ncols), dtype="int32")
    top_cells = [0] * ncols
    if len(moves) - n == -1:
        yield state0
        state0 = state0.copy()
    for i, move0 in enumerate(moves):
        state0[i % 2, top_cells[move0], move0] = 1
        top_cells[move0] += 1
        if i >= len(moves) - n:
            yield state0
            state0 = state0.copy()


def sars((winner0, moves0)):
    nmoves = len(moves0)
    sample_move = -(np.random.randint(0, nmoves - 1) + 1)
    col0 = moves0[sample_move - 1]
    if sample_move == -1:
        state0 = moves_to_states(moves0[:sample_move]).next()
        state1 = None
        if winner0 == 1:
            reward = 1
        elif winner0 == 2:
            reward = -1
        else:
            reward = 0
        return state0, col0, reward, state1
    else:
        state0, state1 = list(moves_to_states(moves0[:sample_move], 2))
        reward = 0
        return state0, col0, reward, state1


def gen_batch(games, n):
    sample_games = [games[i] for i in np.random.randint(len(games), size=n)]
    return map(sars, sample_games)


def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


def get_network(input_var):
    network = lasagne.layers.InputLayer(
        shape=(None, 2, nrows, ncols),
        input_var=input_var,
    )

    network = lasagne.layers.Conv2DLayer(
        network, 32, 4,
    )
    # model.add(
    #     Convolution2D(
    #         32, 4, 4, border_mode="valid",
    #         input_shape=(2, nrows, ncols)
    #     )
    # )
    network = lasagne.layers.DenseLayer(
        network, num_units=8,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    network = lasagne.layers.DenseLayer(
        network, num_units=1,
        nonlinearity=None,
    )
    return network


def compile_functions():
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.vector('targets')

    # Create neural network model (depending on first command line
    # parameter)
    print("Building model and compiling functions...")
    network = get_network(input_var)

    # Create a loss expression for training, i.e., a scalar objective
    # we want to minimize (for our multi-class problem, it is the
    # cross-entropy loss):
    prediction = lasagne.layers.get_output(network).flatten()
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see
    # lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic
    # Gradient Descent (SGD) with Nesterov momentum, but Lasagne
    # offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params)

    # Compile a function performing a training step on a mini-batch
    # (by giving the updates dictionary) and returning the
    # corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    return train_fn, val_fn


games = [parse_game(line0) for line0 in file("RvR.txt").readlines()]

alpha = 0.8

train_fn, val_fn = compile_functions()

# Finally, launch the training loop.
print("Starting training...")

num_epochs = 100
# We iterate over epochs:
for epoch in range(num_epochs):
    state0s, actions, rewards, state1s = zip(*gen_batch(games, 100000))
    state0s = np.array(state0s)

    X_train = state0s
    y_train = np.array(rewards, dtype=theano.config.floatX)

    # In each epoch, we do a full pass over the training data:
    train_err = 0
    train_batches = 0
    start_time = time.time()
    for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
        inputs, targets = batch
        train_err += train_fn(inputs, targets)
        train_batches += 1

    # # And a full pass over the validation data:
    # val_err = 0
    # val_acc = 0
    # val_batches = 0
    # for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
    #     inputs, targets = batch
    #     err = val_fn(inputs, targets)
    #     val_err += err
    #     val_batches += 1

    # Then we print the results for this epoch:
    print("Epoch {} of {} took {:.3f}s".format(
        epoch + 1, num_epochs, time.time() - start_time))
    print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
    # print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))


# network = get_network()
# for i in range(10000):
#     state0s, actions, rewards, state1s = zip(*gen_batch(games, 10000))
#     state0s = np.array(state0s)

    # non_term_indices, non_term_states = zip(
    #     *((i, state1)
    #       for i, state1 in enumerate(state1s)
    #       if state1 is not None)
    # )
    # non_term_indices = np.array(non_term_indices)
    # non_term_states = np.array(non_term_states)

    # oddeven = np.mod(non_term_states.sum(axis=(1, 2, 3)), 2)
    # Q1preds = model.predict(non_term_states)

    # Q1s = np.zeros(len(state1s))
    # Q1s[non_term_indices[oddeven == 0]] = Q1preds[oddeven == 0].max(axis=1)
    # Q1s[non_term_indices[oddeven == 1]] = Q1preds[oddeven == 1].min(axis=1)

    # Q0s = model.predict(np.array(state0s))

    # to_update = zip(*enumerate(actions))

    # Q0s[to_update] = np.array(rewards) + alpha * Q1s

    # print "Iter {0}, Error: {1[0]}".format(
    #     i, model.train_on_batch(state0s, Q0s)
    # )
