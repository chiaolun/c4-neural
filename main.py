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
    if state0 is None:
        return None
    state0 = state0.copy()
    top_cells = state0.max(axis=0).argmin(axis=0)
    side = top_cells.sum() % 2
    for move0 in moves:
        if top_cells[move0] == nrows:
            return None
        state0[side, top_cells[move0], move0] = 1
        top_cells[move0] += 1
        side = 1 - side
    return state0


def get_side(state0):
    return 1 - 2 * (state0.sum() % 2)


def srs((winner0, moves0)):
    no_state = np.zeros((2, nrows, ncols))
    sample_move = np.random.randint(0, len(moves0))
    state0 = moves_to_state(moves0[:sample_move])
    side0 = get_side(state0)
    moves0 = moves0[sample_move:]
    if len(moves0) == 2:
        return state0, -side0, no_state
    elif len(moves0) == 1:
        return state0, side0, no_state
    else:
        state1 = moves_to_state(moves0[:2], state0)
        return state0, 0., state1


def gen_batch(games, n):
    sample_games = [games[i] for i in np.random.randint(len(games), size=n)]
    return map(srs, sample_games)


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
        network, 64, (3, 3),
    )
    network = lasagne.layers.Conv2DLayer(
        network, 64, (3, 3),
    )
    network = lasagne.layers.DenseLayer(
        network, num_units=512,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform()
    )
    network = lasagne.layers.DenseLayer(
        network, num_units=1,
        nonlinearity=None,
    )
    return network


def compile_Q(network):
    state = T.tensor4('state')
    Q = lasagne.layers.get_output(network, inputs=state)
    t_fn = theano.function([state], Q)

    def Q_fn(state0s):
        Qss = []
        for state0 in state0s:
            state1s = [
                moves_to_state([i], state0)
                for i in range(ncols)
            ]
            Qs = np.empty(ncols)
            Qs.fill(np.nan)
            non_zeros = zip(*[
                (i, state1)
                for i, state1 in enumerate(state1s)
                if state1 is not None
            ])
            if len(non_zeros) > 0:
                Qs.flat[list(non_zeros[0])] = t_fn(np.array(non_zeros[1]))
                Qs *= get_side(state0)
            Qss.append(Qs)
        return np.array(Qss)
    return Q_fn


def compile_trainer(network):
    # Prepare Theano variables for inputs and targets
    alpha = T.scalar("alpha")
    state0 = T.tensor4('state0')
    reward = T.vector('reward')
    Q1 = T.vector('state1')

    # Create a loss expression for training, i.e., a scalar objective
    # we want to minimize
    Q0 = lasagne.layers.get_output(network, inputs=state0).flatten()
    # Q1 = lasagne.layers.get_output(network, inputs=state1).flatten()

    error_vec = (
        Q0 - reward - alpha * Q1
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
        [state0, reward, Q1, alpha],
        error,
        updates=updates,
        on_unused_input='warn')

    return train_fn


def save_network(network):
    np.savez("network", *lasagne.layers.get_all_param_values(network))


def load_network():
    network = get_network()

    try:
        with np.load("network.npz") as saved_coefs:
            lasagne.layers.set_all_param_values(
                network,
                [saved_coefs[k].astype(theano.config.floatX)
                 for k in sorted(saved_coefs)]
            )
    except IOError:
        pass

    return network


class network_trainer():

    def __init__(self, network):
        state = T.tensor4('state')
        Q = lasagne.layers.get_output(network, inputs=state)
        self.t_fn = theano.function([state], Q)
        self.train_fn = compile_trainer(network)

    def train(
            self,
            state0s_batch,
            rewards_batch,
            state1s_batch,
            alpha,
    ):
        Q1_batch = self.t_fn(state1s_batch).flatten()
        return self.train_fn(
            state0s_batch,
            rewards_batch,
            Q1_batch,
            alpha,
        )


def main(num_epochs=100):
    games = [parse_game(line0) for line0 in file("RvR.txt").readlines()]

    network = load_network()

    trainer = network_trainer(network)

    # Finally, launch the training loop.
    print("Starting training...")

    # We iterate over epochs:
    for epoch in range(num_epochs):
        start_time = time.time()
        state0s, rewards, state1s = zip(*gen_batch(games, 50000))
        state0s = np.array(state0s, dtype=theano.config.floatX)
        rewards = np.array(rewards, dtype=theano.config.floatX)
        state1s = np.array(state1s, dtype=theano.config.floatX)

        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        for (
                state0s_batch,
                rewards_batch,
                state1s_batch,
        ) in iterate_minibatches(
                state0s, rewards, state1s,
                batchsize=500, shuffle=True,
        ):
            train_err += trainer.train(
                state0s_batch,
                rewards_batch,
                state1s_batch,
                0.9
            )
            train_batches += 1

        # Save coefficients
        save_network(network)

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))


# Q_fn(np.array([flip_state(moves_to_state([3,3,3,2,2,2]))])).argmax()

if __name__ == "__main__":
    main(100)
