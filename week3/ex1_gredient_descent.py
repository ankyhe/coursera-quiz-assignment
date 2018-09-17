import numpy as np

from utils import computer_cost, sigmoid, load_data, plot_data


def gradient_descent(x, y, alpha, iteration):
    m = y.size
    theta = np.random.rand(3, 1)
    j_list = []
    for i in range(iteration):
        v = computer_cost.compute2(x, y, theta)
        j_list.append(v)
        values = (sigmoid.sigmoid(x.dot(theta)) - y).T.dot(x).T * (alpha / m)
        theta = theta - values
    return theta, np.array(j_list)


def main():
    x, y, origin_x = load_data.load_and_process('data1.txt')
    alpha = 0.001
    iterations = 100
    theta, j_list = gradient_descent(x, y, alpha, iterations)
    print(theta)

    '''
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
    '''

    iteration_array = range(iterations)
    plot_data.plot(iteration_array, j_list, 'iteration', "lost",
                   {
                       'fmt': 'b-',
                       'title': 'Lost',
                       'show': True
                   })


if __name__ == '__main__':
    main()