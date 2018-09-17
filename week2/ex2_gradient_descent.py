import numpy as np

from utils import load_data, plot_data, computer_cost, normalize


def gradient_descent(X, Y, alpha, iteration):
    m = Y.size
    theta = np.zeros((X.shape[1], 1))
    j_list = []
    for i in range(iteration):
        v = computer_cost.computer(X, Y, theta)
        j_list.append(v)
        values = (X.dot(theta) - Y).T.dot(X).T * (alpha / m)
        theta = theta - values
    return theta, np.array(j_list)


def main():
    X_normalize, Y_normalize, Origin_X_normalize, Y_mean, Y_std, Origin_X_mean, Origin_X_std = load_data.load_and_normalize('data2.txt')

    alpha = 0.024
    iterations = 800
    theta, j_list = gradient_descent(X_normalize, Y_normalize, alpha, iterations)
    print('The linear regression formula is {0} + {1} * x1 + {2} * x2'.format(theta[0, 0], theta[1, 0], theta[2,0]))

    iteration_array = range(iterations)
    plot_data.plot(iteration_array, j_list, 'iteration', "lost",
                   {
                       'fmt': 'b-',
                       'title': 'Lost',
                       'show': True
                   })

    house_size = 1650
    br_num = 3

    house_size_normal = (house_size - Origin_X_mean[0]) / Origin_X_std[0]
    br_num = (br_num - Origin_X_mean[1]) / Origin_X_std[1]

    X_new_normalize = np.array([1, house_size_normal, br_num])
    price_normalize = X_new_normalize.dot(theta).sum()
    price = price_normalize * Y_std[0] + Y_mean[0]
    print('price is {0}'.format(price))


if __name__ == '__main__':
    main()
