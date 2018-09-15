import numpy as np


def computer(X, Y, theta):
    m = Y.size
    lost = (X.dot(theta) - Y)
    return np.power(lost, 2).sum() / (2.0 * m)
