import numpy as np

from utils import sigmoid


def computer(X, Y, theta):
    m = Y.size
    lost = (X.dot(theta) - Y)
    return np.power(lost, 2).sum() / (2.0 * m)


def compute2(x, y, theta):
    m = y.size

    a = 1 - sigmoid.sigmoid(x.dot(theta))
    b = sigmoid.sigmoid(x.dot(theta))

    value = x.dot(theta)
    v = 1.0
    for idx, item in enumerate(y):
        if item == 0:
            v *= sigmoid.sigmoid(value[idx, 0])
        else:
            v *= (1 - sigmoid.sigmoid(value[idx, 0]))

    return -np.log(v) / m
