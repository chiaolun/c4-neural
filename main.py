#!/usr/bin/env python

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
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


def construct_model(nlayers, optimizer="sgd"):
    model = Sequential()
    for i in range(nlayers):
        model.add(
            Dense(
                output_dim=state_dim * 2,
                init='lecun_uniform',
                input_dim=2 * state_dim if i == 0 else None
            )
        )
        model.add(Activation('relu'))
        # model.add(Dropout(0.5))
    model.add(Dense(3, init='lecun_uniform'))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

games = [parse_game(line0) for line0 in file("RvR.txt").readlines()]
y_raw, X_raw = zip(*games)
y_raw = np.array(y_raw)

censor = np.random.randint(0, 2, len(X_raw))

X_all = np.array([moves_to_state(moves0[:(-n if n > 0 else None)])
                  for n, moves0 in zip(censor, X_raw)])
# X_all = np.array([moves_to_state(moves0) for moves0 in X_raw])
y_all = np.array([(y_raw == i) for i in range(3)]).T * 1

X_train, X_test, y_train, y_test, censor_train, censor_test = train_test_split(
    X_all, y_all, censor, test_size=1000,
    random_state=42,
)

# sgd = SGD(lr=0.1, decay=0, momentum=0.1, nesterov=True)
model0 = construct_model(5, optimizer="adam")
model0.fit(
    X_train, y_train,
    nb_epoch=5,
    validation_data=(X_test, y_test),
    show_accuracy=True,
)
