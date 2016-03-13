#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.core import Dense, Activation
import numpy as np
from sklearn.cross_validation import train_test_split

ncols = 7
nrows = 6
state_dim = ncols * nrows

# http://outlace.com/Reinforcement-Learning-Part-3/


def parse_game(line0):
    line0 = [int(char0) for char0 in line0.strip()]
    winner = line0[0]
    moves = line0[1:]
    return winner, moves


games = [parse_game(line0) for line0 in file("RvR.txt").readlines()]


def moves_to_state(moves):
    state0 = [0] * state_dim
    top_cells = [nrows * i for i in range(ncols)]
    side = 1
    for move0 in moves:
        state0[top_cells[move0]] = side
        side = {1: 2, 2: 1}[side]
        top_cells[move0] += 1
    return state0


y_all, X_all = zip(*games)
y_all = np.array(y_all)
X_all = np.array([moves_to_state(moves0) for moves0 in X_all])
X_all = np.concatenate([(X_all == i) for i in [1, 2]], axis=1) * 1
y_all = np.array([(y_all == i) for i in range(3)]).T * 1

X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, train_size=10000, test_size=10000, random_state=42
)


def construct_model(nlayers=2, optimizer="sgd"):
    model = Sequential()
    for i in range(nlayers):
        model.add(
            Dense(
                output_dim=state_dim * 2 * 10,
                init='lecun_uniform',
                input_dim=state_dim * 2 if i == 0 else None
            )
        )
        model.add(Activation('relu'))
        # model.add(Dropout(0.5))
    model.add(Dense(3, init='lecun_uniform'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

# sgd = SGD(lr=0.1, decay=0, momentum=0.1, nesterov=True)
model0 = construct_model(2, optimizer="sgd")
model0.fit(
    X_train, y_train, nb_epoch=200,
    validation_data=(X_test, y_test),
    show_accuracy=True,
)
