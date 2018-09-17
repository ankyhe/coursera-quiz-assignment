import numpy as np

from utils import load_data, plot_data


def main():
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
                       'show': True
                   })


if __name__ == '__main__':
    main()

