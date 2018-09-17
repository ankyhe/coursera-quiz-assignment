import numpy as np


def normalize(X):
    mean = np.mean(X, axis=0)
    std_deviation = np.std(X, axis=0)
    return (X - mean) / (std_deviation), mean, std_deviation

