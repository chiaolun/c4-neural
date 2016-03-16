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


def moves_to_states(moves, n=1):
    state0 = np.zeros((2, nrows, ncols))
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
    model.add(Dense(ncols, init='lecun_uniform'))
    model.add(Activation('linear'))
    model.compile(loss='mse', optimizer="adam")
    return model


def sars((winner0, moves0)):
    nmoves = len(moves0)
    sample_move = -(np.random.randint(0, nmoves - 1) + 1)
    col0 = moves0[sample_move - 1]
    if sample_move == -1:
        state0 = moves_to_states(moves0[:sample_move]).next()
        state1 = None
        if winner0 == 1:
            reward = 1
        elif winner0 == 0:
            reward = 0.5
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

games = [parse_game(line0) for line0 in file("RvR.txt").readlines()]

alpha = 0.8

model = get_model()
for i in range(1000000):
    state0s, actions, rewards, state1s = zip(*gen_batch(games, 10000))
    state0s = np.array(state0s)

    non_term_indices, non_term_states = zip(
        *((i, state1)
          for i, state1 in enumerate(state1s)
          if state1 is not None)
    )
    non_term_indices = np.array(non_term_indices)
    non_term_states = np.array(non_term_states)

    Q1s = np.zeros(len(state1s))
    Q1s[non_term_indices] = model.predict(non_term_states).max(axis=1)

    Q0s = model.predict(np.array(state0s))

    to_update = zip(*enumerate(actions))

    Q0s[to_update] = np.array(rewards) + alpha * Q1s

    print "Iter {0}, Error: {1[0]}".format(
        i, model.train_on_batch(state0s, Q0s)
    )
