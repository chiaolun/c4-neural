#!/usr/bin/env python

import numpy as np
from sklearn.cross_validation import train_test_split

import time

import theano
import theano.tensor as T

import lasagne

# Code copied from lasagne:
# https://github.com/Lasagne/Lasagne/blob/5fcc4aa80fc9a299fbe444a2fa490333a8af6142/LICENSE

ncols = 7
nrows = 6
state_dim = ncols * nrows


def parse_game(line0):
    line0 = [int(char0) for char0 in line0.strip()]
    winner = line0[0]
    moves = line0[1:]
    return winner, moves


def moves_to_state(moves):
    state0 = np.zeros(state_dim * 2)
    top_cells = [nrows * i for i in range(ncols)]
    side = 0
    for move0 in moves:
        state0[top_cells[move0] * 2 + side] = 1
        side = (2 - side) // 2
        top_cells[move0] += 1
    return state0


def censor_data(ncensor, X_raw, y_raw):
    censor = np.random.randint(0, ncensor + 1, len(X_raw))
    X_all = np.array([
        moves_to_state(moves0[:(-n if n > 0 else None)])
        for n, moves0 in zip(censor, X_raw)
    ], dtype=theano.config.floatX)
    y_all = y_raw

    (
        X_train,
        X_val,
        y_train,
        y_val,
        # censor_train,
        # censor_val
    ) = train_test_split(
        X_all, y_all,  # censor,
        test_size=10000,
        random_state=42,
    )
    return X_train, X_val, y_train, y_val


def get_network(input_var):
    network = lasagne.layers.InputLayer(
        shape=(None, state_dim * 2),
        input_var=input_var,
    )
    network = lasagne.layers.DenseLayer(
        network, num_units=800,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    network = lasagne.layers.DenseLayer(
        network, num_units=3,
        nonlinearity=lasagne.nonlinearities.softmax
    )
    return network


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


def main(num_epochs=500):
    # Load the dataset
    print("Loading data...")

    games = [parse_game(line0) for line0 in file("RvR.txt").readlines()]
    y_raw, X_raw = zip(*games)
    y_raw = np.array(y_raw, dtype="int32")

    (
        X_train,
        X_val,
        y_train,
        y_val,
    ) = censor_data(3, X_raw, y_raw)

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line
    # parameter)
    print("Building model and compiling functions...")
    network = get_network(input_var)

    # Create a loss expression for training, i.e., a scalar objective
    # we want to minimize (for our multi-class problem, it is the
    # cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see
    # lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic
    # Gradient Descent (SGD) with Nesterov momentum, but Lasagne
    # offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.adam(loss, params)

    # Create a loss expression for validation/testing. The crucial
    # difference here is that we do a deterministic forward pass
    # through the network, disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification
    # accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch
    # (by giving the updates dictionary) and returning the
    # corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and
    # accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

main(5)
