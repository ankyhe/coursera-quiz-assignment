import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt


from utils import sigmoid, load_data, plot_data, feature_map

epsilon = 1e-5


def cost_function_reg(theta, reg, *args):
    #print('args\n{0}\naaa'.format(args))
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


def predict(theta, X, threshold=0.5):
    p = sigmoid.sigmoid(X.dot(theta.T)) >= threshold
    return(p.astype('int'))


def plotData(data, label_x, label_y, label_pos, label_neg, axes=None):
    # Get indexes for class 0 and class 1
    neg = data[:, 2] == 0
    pos = data[:, 2] == 1

    # If no specific axes object has been passed, get the current axes.
    if axes == None:
        axes = plt.gca()
    axes.scatter(data[pos][:, 0], data[pos][:, 1], marker='+', c='k', s=60, linewidth=2, label=label_pos)
    axes.scatter(data[neg][:, 0], data[neg][:, 1], c='y', s=60, label=label_neg)
    axes.set_xlabel(label_x)
    axes.set_ylabel(label_y)
    axes.legend(frameon=True, fancybox=True);


def main():
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(17, 5))
    data = load_data.load('data2.txt', dtype = np.float128)
    X = data[:, 0:2]
    X_map = feature_map.map(X)
    y = data[:, 2].reshape(-1, 1)
    initial_theta = np.zeros(X_map.shape[1])
    #C = 0
    #res = minimize(cost_function_reg, initial_theta, args=(C, X_map, y), method=None, jac=gradient_reg, options={'maxiter': 3000})
    #print(res)
    for i, C in enumerate([0, 1, 100]):
        # Optimize costFunctionReg
        res2 = minimize(cost_function_reg, initial_theta, args=(C, X_map, y), method=None, jac=gradient_reg,options={'maxiter': 3000})
        accuracy = 100 * sum(predict(res2.x, X_map) == y.ravel()) / y.size
        plotData(data, 'Microchip Test 1', 'Microchip Test 2', 'y = 1', 'y = 0', axes.flatten()[i])
        # Plot decisionboundary
        x1_min, x1_max = X[:, 0].min(), X[:, 0].max(),
        x2_min, x2_max = X[:, 1].min(), X[:, 1].max(),
        xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
        h = sigmoid.sigmoid(feature_map.map(np.c_[xx1.ravel(), xx2.ravel()]).dot(res2.x.reshape(-1, 1)))
        h = h.reshape(xx1.shape)
        axes.flatten()[i].contour(xx1, xx2, h, [0.5], linewidths=1, colors='g');
        axes.flatten()[i].set_title('Train accuracy {}% with Lambda = {}'.format(np.round(accuracy, decimals=2), C))

    plt.show()


if __name__ == '__main__':
    main()
