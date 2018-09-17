import numpy as np


def sigmoid_theory(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid(x):
    try:
        iter(x)
    except TypeError:
        return sigmoid_theory(x)
    else:
        ret = []
        for item in x:
            if item >= 0:
                z = np.exp(-item)
                ret.append(1 / (1 + z))
            else:
                z = np.exp(item)
                ret.append(z / (1 + z))
        return np.array(ret)

