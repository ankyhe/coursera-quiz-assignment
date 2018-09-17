import numpy as np

from utils import computer_cost, sigmoid, load_data, plot_data


def gradient_descent(x, y, alpha, iteration):
    m = y.size
    theta = np.zeros(3).reshape(3, -1)
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
    iterations = 1000000
    theta, j_list = gradient_descent(x, y, alpha, iterations)
    print(theta)

    iteration_array = range(iterations)
    plot_data.plot(iteration_array, j_list, 'iteration', "lost",
                   {
                       'fmt': 'b-',
                       'title': 'Lost',
                       'show': True
                   })

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
        score2.append(((0.5 - theta[0, 0]) - theta[1, 0] * item) / theta[2, 0])
    score2 = np.array(score2)

    plot_data.plot(score1, score2, 'score1', 'score2',
                   {
                       'fmt': 'r-',
                       'show': True
                   })


if __name__ == '__main__':
    main()