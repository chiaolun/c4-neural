#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.callbacks import EarlyStopping
import numpy as np
import itertools
from sklearn.cross_validation import train_test_split
import cPickle

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


def construct_model(layers, optimizer, activation, dropout):
    model = Sequential()
    for layer0 in layers:
        model.add(layer0)
        model.add(Activation(activation))
        if dropout:
            model.add(Dropout(dropout))
    model.add(Dense(3, init='lecun_uniform'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


def censor_data(ncensor):
    censor = np.random.randint(0, ncensor + 1, len(X_raw))
    X_all = np.array([moves_to_state(moves0[:(-n if n > 0 else None)])
                      for n, moves0 in zip(censor, X_raw)])
    # X_all = np.array([moves_to_state(moves0) for moves0 in X_raw])
    y_all = np.array([(y_raw == i) for i in range(3)]).T * 1

    (
        X_train,
        X_test,
        y_train,
        y_test,
        censor_train,
        censor_test
    ) = train_test_split(
        X_all, y_all, censor,
        train_size=100000, test_size=10000,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test, censor_train, censor_test


def linear_layers(n, flat):
    if flat:
        sizes = [state_dim * 2 for _ in range(n + 1)]
    else:
        sizes = np.linspace(state_dim * 2, 3, n + 1).round().astype("int")
    return [
        Dense(
            output_dim=n0,
            init='lecun_uniform',
            input_dim=2 * state_dim if i == 0 else None
        ) for i, n0 in enumerate(sizes[:-1])
    ]


games = [parse_game(line0) for line0 in file("RvR.txt").readlines()]
y_raw, X_raw = zip(*games)
y_raw = np.array(y_raw)

ncensors = [1, 2, 3, 4, 5]
optimizers = ["sgd", "adam", "adadelta"]
activations = ["relu", "tanh"]
nlayers = [1, 3, 5, 8, 10]
trainsizes = [1000, 10000, 100000]
dropouts = [None, 0.1, 0.5]
flats = [True, False]

results = {}
try:
    results = cPickle.load(file("results.pickle", "w"))
except:
    pass

for ncensor0 in ncensors:
    (
        X_train,
        X_test,
        y_train,
        y_test,
        censor_train,
        censor_test
    ) = censor_data(ncensor0)
    for (
            optimizer0,
            activation0,
            dropout0,
            nlayer0,
            flat0,
            trainsize0
    ) in itertools.product(
        optimizers,
        activations,
        dropouts,
        nlayers,
        flats,
        trainsizes
    ):
        params0 = dict(
            ncensor=ncensor0,
            optimizer=optimizer0,
            activation=activation0,
            dropout=dropout0,
            nlayer=nlayer0,
            flat=flat0,
            trainsize=trainsize0,
        )
        if tuple(sorted(params0.items())) in results:
            print "Skipping params {}".format(params0)
            continue
        print "Testing params {}".format(params0)

        layers = linear_layers(nlayer0, flat=flat0)
        model0 = construct_model(
            layers,
            optimizer=optimizer0,
            activation=activation0,
            dropout=dropout0,
        )
        data0 = model0.fit(
            X_train[:trainsize0],
            y_train[:trainsize0],
            batch_size=128,
            nb_epoch=10000000 // trainsize0,
            validation_data=(X_test, y_test),
            show_accuracy=True,
            callbacks=[EarlyStopping(patience=10)],
        ).history
        results[tuple(sorted(params0.items()))] = data0
        cPickle.dump(results, file("results.pickle", "w"))
