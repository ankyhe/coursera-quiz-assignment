import numpy as np
import matplotlib.pyplot as plt

from utils import load_data, plot_data, computer_cost


def gradient_descent(X, Y, alpha, iteration):
    m = Y.size
    theta = np.zeros((2, 1))
    j_list = []
    for i in range(iteration):
        v = computer_cost.computer(X, Y, theta)
        j_list.append(v)
        values = (X.dot(theta) - Y).T.dot(X).T * (alpha / m)
        theta = theta - values
    return theta, np.array(j_list)


def main():
    X, Y, Origin_X = load_data.load_and_process('data1.txt')
    alpha = 0.023
    iterations = 400
    theta, j_list = gradient_descent(X, Y, alpha, iterations)
    print(theta)

    plot_data.plot(Origin_X, Y, 'original data figure', 'house size',
                   {
                       'fmt': 'rx',
                       'label': 'Original Data'
                   })
    plot_data.plot(Origin_X, X.dot(theta), "house size", 'price',
                   {
                       'fmt': 'b-',
                       'title': 'gradient decent',
                       'label': 'Linear Regression',
                       'legend_loc': 'lower right',
                       'show': True
                   })

    iteration_array = range(iterations)
    plot_data.plot(iteration_array, j_list, 'iteration', "lost",
                   {
                       'fmt': 'b-',
                       'title': 'Lost',
                       'show': True
                   })


if __name__ == '__main__':
    main()
