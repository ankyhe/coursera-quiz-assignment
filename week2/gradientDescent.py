import numpy as np

import computeCost as cc

def gradientDescent(X, Y, theta, alpha, num_iters):

    m = np.size(Y)
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        theta = theta - alpha*(1.0/m) * np.transpose(X).dot(X.dot(theta) - Y)
        J_history[i] = cc.computeCost(X, Y, theta)

    return theta, J_history