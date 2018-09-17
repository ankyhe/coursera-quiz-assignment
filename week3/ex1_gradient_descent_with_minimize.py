import numpy as np
from scipy.optimize import minimize

from utils import computer_cost, sigmoid, load_data, plot_data

epsilon = 1e-5


def cost_function(theta, X, y):
    m = y.size
    h = sigmoid.sigmoid(X.dot(theta))
    J = -1 * (1 / m) * (np.log(h + epsilon).T.dot(y) + np.log(1 - h + epsilon).T.dot(1 - y))
    return J[0]


def gradient(theta, X, y):
    m = y.size
    h = sigmoid.sigmoid(X.dot(theta.reshape(-1, 1)))
    grad = (1 / m) * X.T.dot(h - y)
    return grad.flatten()


def predict(score1, score2, theta):
    return sigmoid.sigmoid(np.array([1, score1, score2]).dot(theta.reshape(3, -1)).sum())


def main():
    X, y, origin_X = load_data.load_and_process('data1.txt')
    initial_theta = np.zeros(X.shape[1])
    res = minimize(cost_function, initial_theta, args=(X, y), method=None, jac=gradient, options={'maxiter': 400})
    print(res)

    theta = res.x

    data = load_data.load('data1.txt')

    pos_values = data[(data[:, 2] == 1)]
    neg_values = data[(data[:, 2] == 0)]

    plot_data.plot(pos_values[:, 0], pos_values[:, 1], 'score1', 'score2',
                   {
                       'fmt': 'bx',
                       'markersize': 5
                   })

    plot_data.plot(neg_values[:, 0], neg_values[:, 1], 'score1', 'score2',
                   {
                       'fmt': 'yo',
                       'markersize': 5,
                       'show': False
                   })

    score1 = np.linspace(25, 100)
    score2 = []
    for item in score1:
        score2.append(((0.5 - theta[0]) - theta[1] * item) / theta[2])
    score2 = np.array(score2)

    plot_data.plot(score1, score2, 'score1', 'score2',
                   {
                       'fmt': 'r-',
                       'label': 'minimize',
                       'show': False
                   })

    theta_from_cal = np.array([-4.81180027, 0.04528064, 0.03819149]) # 0.001 10 0000
    theta_from_cal_2 = np.array([-15.39517866, 0.12825989, 0.12247929]) # 0.001 100 0000
    theta_from_cal_3 = np.array([-22.21628108, 0.18268725, 0.17763448]) # 0.003 100 0000
    score3 = []
    for item in score1:
        score3.append(((0.5 - theta_from_cal_3[0]) - theta_from_cal_3[1] * item) / theta_from_cal_3[2])
    score3 = np.array(score3)
    plot_data.plot(score1, score3, 'score1', 'score2',
                   {
                       'fmt': 'g-',
                       'label': 'gredient-descent',
                       'legend_loc': 'upper right',
                       'show': True
                   })

    print(predict(45, 85, theta_from_cal_3))


if __name__ == '__main__':
    main()

