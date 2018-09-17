import numpy as np

from utils import sigmoid

epsilon = 1e-5

def computer(X, Y, theta):
    m = Y.size
    lost = (X.dot(theta) - Y)
    return np.power(lost, 2).sum() / (2.0 * m)


def compute2(x, y, theta):
    m = y.size
    h = sigmoid.sigmoid(x.dot(theta))
    j = -1 * (1 / m) * (np.log(h + epsilon).T.dot(y) + np.log(1 - h + epsilon).T.dot(1 - y))
    return j[0]


def compute3(x, y, theta):
    m = y.size
    value = x.dot(theta)
    v = 1.0
    for idx, item in enumerate(y):
        if item == 1:
            v *= sigmoid.sigmoid(value[idx, 0])
        else:
            v *= (1 - sigmoid.sigmoid(value[idx, 0]))

    ret = -np.log(v) / m
    if np.isnan(ret):
        return np.inf
    return ret
