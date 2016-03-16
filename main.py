#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Reshape
from keras.layers import Convolution2D
import numpy as np

ncols = 7
nrows = 6


def parse_game(line0):
    line0 = [int(char0) for char0 in line0.strip()]
    winner = line0[0]
    moves = line0[1:]
    return winner, moves


def moves_to_state(moves):
    state0 = np.zeros((2, nrows, ncols))
    top_cells = [0] * ncols
    state0 = state0.copy()
    for i, move0 in enumerate(moves):
        state0[i % 2, top_cells[move0], move0] = 1
        top_cells[move0] += 1
    return state0


def moves_to_states(moves):
    state0 = np.zeros((2, nrows, ncols))
    top_cells = [0] * ncols
    yield state0
    state0 = state0.copy()
    for i, move0 in enumerate(moves):
        state0[i % 2, top_cells[move0], move0] = 1
        top_cells[move0] += 1
        if i < len(moves) - 1:
            yield state0
            state0 = state0.copy()
    yield state0


def rstates((winner0, moves0)):
    states = list(moves_to_states(moves0))
    if winner0 == 1:
        reward = 1
    elif winner0 == 0:
        reward = 0.5
    else:
        reward = 0

    for state0 in states:
        yield reward, state0


def get_model():
    model = Sequential()
    model.add(
        Convolution2D(
            32, 4, 4, border_mode="valid",
            input_shape=(2, nrows, ncols)
        )
    )
    model.add(Activation("relu"))
    model.add(Flatten())
    model.add(Dense(8, init='lecun_uniform'))
    model.add(Activation('relu'))
    model.add(Dense(1, init='lecun_uniform'))
    model.add(Activation('sigmoid'))
    model.compile(loss='binary_crossentropy',
                  optimizer="adam",
                  class_mode="binary")
    return model


def rstate_batch(games, n):
    results = []
    while 1:
        for game0 in games:
            for line0 in rstates(game0):
                results.append(line0)
                if len(results) >= n:
                    r, s = zip(*results)
                    yield np.array(s), np.array(r)
                    results = []

games = [parse_game(line0) for line0 in file("RvR.txt").readlines()]

model = get_model()

for X, y in rstate_batch(games, 100000):
    model.fit(X, y, nb_epoch=1, show_accuracy=True)
