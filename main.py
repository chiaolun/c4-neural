#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.advanced_activations import PReLU
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.cross_validation import train_test_split

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


def construct_model(layers, optimizer, activation):
    model = Sequential()
    for layer0 in layers:
        model.add(layer0)
        if activation == "prelu":
            model.add(PReLU())
        else:
            model.add(Activation(activation))
    model.add(Dense(3, init='lecun_uniform'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


def censor_data(ncensor, X_raw, y_raw):
    censor = np.random.randint(0, ncensor + 1, len(X_raw))
    X_all = np.array([
        moves_to_state(moves0[:(-n if n > 0 else None)])
        for n, moves0 in zip(censor, X_raw)
    ])
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
        test_size=10000,
        random_state=42,
    )
    return X_train, X_test, y_train, y_test, censor_train, censor_test


def linear_layers(n):
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

(
    X_train,
    X_test,
    y_train,
    y_test,
    censor_train,
    censor_test
) = censor_data(3, X_raw, y_raw)
layers = linear_layers(5)
model0 = construct_model(
    layers,
    optimizer="adam",
    activation="relu",
)
model0.fit(
    X_train,
    y_train,
    batch_size=128,
    nb_epoch=5,
    validation_data=(X_test, y_test),
    show_accuracy=True,
    callbacks=[EarlyStopping(patience=10)],
)
