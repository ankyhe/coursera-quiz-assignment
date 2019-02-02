import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from utils import load_data
from utils import plot_data

def normalize_by_self():
    X_normalize, Y_normalize, Origin_X_normalize, Y_mean, Y_std, Origin_X_mean, Origin_X_std = load_data.load_and_normalize('data2.txt')
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(X_normalize, Y_normalize)
    print(linear_regression)
    house_size = 1650
    br_num = 3
    house_size_normalize = (house_size - Origin_X_mean[0]) / Origin_X_std[0]
    br_num_normalize = (br_num - Origin_X_mean[1]) / Origin_X_std[1]
    price_normalize = linear_regression.predict(np.array([1, house_size_normalize, br_num_normalize]).reshape(1, 3))
    price = price_normalize * Y_std[0] + Y_mean[0]
    print('house prices is {0}'.format(price))

def main():
    X, Y, X_Original = load_data.load_and_process('data2.txt')
    linear_regression = linear_model.LinearRegression()
    linear_regression.fit(X, Y)
    print('should normalize? {0}'.format(linear_regression.normalize))
    house_size = 1650
    br_num = 3
    price = linear_regression.predict(np.array([1, house_size, br_num]).reshape(1, 3))
    print('house prices is {0}'.format(price))


if __name__ == '__main__':
    warnings.filterwarnings(action="ignore", module="sklearn", message="^internal gelsd")
    main()