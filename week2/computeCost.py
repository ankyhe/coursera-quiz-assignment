import numpy as np

def computeCost(X, Y, theta):
    m = np.size(Y)
    s = np.power((X.dot(theta) - Y), 2)
    return (1.0 / (2 * m)) * s.sum(axis = 0)
