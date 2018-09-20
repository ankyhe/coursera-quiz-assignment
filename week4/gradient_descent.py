import numpy as np
from scipy.optimize import minimize

from utils import sigmoid, load_data


epsilon = 1e-5


def cost_function_reg(theta, reg, *args):
    y = args[1]
    X = args[0]
    m = y.size
    h = sigmoid.sigmoid(X.dot(theta))
    J = -1 * (1 / m) * (np.log(h + epsilon).T.dot(y) + np.log(1 - h + epsilon).T.dot(1 - y)) + (reg/(2*m))*np.sum(np.square(theta[1:]))
    return J[0]


def gradient_reg(theta, reg, *args):
    y = args[1]
    X = args[0]
    m = y.size
    h = sigmoid.sigmoid(X.dot(theta.reshape(-1, 1)))
    grad = (1 / m) * X.T.dot(h - y) + (reg/m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
    return grad.flatten()


def calc(n_labels, reg, X, y):
    initial_theta = np.zeros((X.shape[1], 1))
    all_theta = np.zeros((n_labels, X.shape[1]))

    for C in np.arange(1, n_labels + 1):
        result = minimize(cost_function_reg, initial_theta, args=(reg, X, (y==C) * 1), method=None, jac=gradient_reg, options={'maxiter': 400})
        all_theta[C - 1] = result.x
    return all_theta


def predict(all_theta, X_new):
    values = X_new.dot(all_theta.T) # (1, 10)
    return np.argmax(values, axis=1) + 1


def main():
    data = load_data.load_mat('ex3data1.mat')
    X_Orign = data['X']
    y = data['y']
    ones = np.ones((X_Orign.shape[0], 1))
    X = np.c_[ones, X_Orign]
    print('X and y shapes are {0} {1}'.format(X.shape, y.shape))
    all_theta = calc(10, 0.2, X, y)
    predict_values = predict(all_theta, X)
    print('Training set accuracy: {} %'.format(np.mean(predict_values == y.ravel()) * 100))


if __name__ == '__main__':
    main()