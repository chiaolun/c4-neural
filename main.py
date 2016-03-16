#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Convolution2D
from keras.callbacks import EarlyStopping
import numpy as np
from sklearn.cross_validation import train_test_split

ncols = 7
nrows = 6


def parse_game(line0):
    line0 = [int(char0) for char0 in line0.strip()]
    winner = line0[0]
    moves = line0[1:]
    return winner, moves


def moves_to_state(moves):
    state0 = np.zeros((2, nrows, ncols))
    top_cells = [0 for _ in range(ncols)]
    side = 0
    for move0 in moves:
        state0[side, top_cells[move0], move0] = 1
        side = (2 - side) // 2
        top_cells[move0] += 1
    return state0


def construct_model(nlayers):
    model = Sequential()
    for i in range(nlayers):
        model.add(
            Convolution2D(
                32, 4, 4, border_mode="valid",
                input_shape=(2, nrows, ncols) if i == 0 else None
            )
        )
        model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(64, init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(1, init='lecun_uniform'))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  class_mode="binary")
    return model


def censor_data(ncensor, X_raw, y_raw):
    censor = np.random.randint(0, ncensor + 1, len(X_raw))
    X_all = np.array([
        moves_to_state(moves0[:(-n if n > 0 else None)])
        for n, moves0 in zip(censor, X_raw)
    ])
    # y_all = np.array([(y_raw == i) for i in range(3)]).T * 1
    y_all = y_raw

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
        # random_state=42,
    )
    return X_train, X_test, y_train, y_test, censor_train, censor_test


games = [parse_game(line0) for line0 in file("RvR.txt").readlines()]
y_raw, X_raw = zip(*games)
y_raw = np.array(y_raw)
# y_raw = np.select([y_raw == 1, y_raw == 0], [1., 0.5])
y_raw = (y_raw == 1) * 1

(
    X_train,
    X_test,
    y_train,
    y_test,
    censor_train,
    censor_test
) = censor_data(1, X_raw, y_raw)

model0 = construct_model(5)

model0.fit(
    X_train,
    y_train,
    batch_size=128,
    nb_epoch=100,
    validation_data=(X_test, y_test),
    show_accuracy=True,
    callbacks=[EarlyStopping(patience=10)],
)
