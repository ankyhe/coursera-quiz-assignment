import numpy as np
from scipy.optimize import minimize

from utils import sigmoid, load_data


def load():
    data = load_data.load_mat('ex3data1.mat')
    X_Origin = data['X']
    ones = np.ones((X_Origin.shape[0], 1))
    X = np.c_[ones, X_Origin]
    y = data['y']
    weight = load_data.load_mat('ex3weights.mat')
    theta1 = weight['Theta1']
    theta2 = weight['Theta2']
    print('theta1 and theta2 are {0} and {1}'.format(theta1.shape, theta2.shape))
    return X, y, theta1, theta2


def predict(theta1, theta2, input_layer):
    layer2 = sigmoid.sigmoid(theta1.dot(input_layer)) # (25, 401) * (401, 5000) = (25 * 5000)
    layer2_new = np.r_[np.ones((1, input_layer.shape[1])), layer2] # (26 * 5000)
    output_layer = sigmoid.sigmoid(theta2.dot(layer2_new)) # (10, 26) * (26 * 5000) = (10 * 5000)
    return np.argmax(output_layer, axis=0) + 1


def main():
    X, y, theta1, theta2 = load()
    predict_values = predict(theta1, theta2, X.T)  # (1 * 5000)
    print('Training set accuracy: {} %'.format(np.mean(predict_values == y.ravel()) * 100))


if __name__ == '__main__':
    main()