import numpy as np


def normal_eqn(X, Y):
    a = X.T.dot(X)
    theta = np.linalg.pinv(a).dot(X.T).dot(Y)
    return theta

