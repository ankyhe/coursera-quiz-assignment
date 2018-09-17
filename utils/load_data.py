import numpy as np
import os

from utils import normalize


def load(filename, print_original_data = False):
    file_path = os.path.join('data', filename)
    ret = np.loadtxt(file_path, delimiter=',')
    if print_original_data:
        print('raw data from file is\n{0}'.format(ret))
    return ret


def load_and_process(filename):
    data = load(filename)
    columns = data.shape[1]
    Y = data[:, columns - 1]
    Y = Y.reshape(-1, 1)
    m = Y.size
    ones = np.ones(m)
    X = data[:, 0 : columns - 1]
    return np.column_stack((ones, X)), Y, X


def load_and_normalize(filename):
    data = load(filename)
    columns = data.shape[1]
    Y = data[:, columns - 1]
    Y = Y.reshape(-1, 1)
    m = Y.size
    ones = np.ones(m)
    X = data[:, 0 : columns - 1]

    X_normalize, X_mean, X_std = normalize.normalize(X)
    Y_normalize, Y_mean, Y_std = normalize.normalize(Y)
    return np.column_stack((ones, X_normalize)), Y_normalize, X_normalize, Y_mean, Y_std, X_mean, X_std
