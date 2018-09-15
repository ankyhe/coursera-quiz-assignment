import numpy as np
import os


def load(filename, print_original_data = False):
    file_path = os.path.join('data', filename)
    ret = np.loadtxt(file_path, delimiter=',')
    if print_original_data:
        print('raw data from file is\n{0}'.format(ret))
    return ret


def load_and_process(filename):
    data = load(filename)
    Y = data[:, 1]
    Y = Y.reshape(-1, 1)
    m = Y.size
    ones = np.ones(m)
    X = data[:, 0]
    return np.column_stack((ones, X)), Y, X
