import numpy as np

from utils import load_data, normalize


def main():
    data = load_data.load('data2.txt')
    X = data[:, 0:2]
    print('X is \n{0}\n'.format(X))
    print('mean(X) is {0}'.format(np.mean(X, axis=0)))
    print('std(X) is {0}'.format(np.std(X, axis=0)))
    print('normalization of X is\n {0}'.format(normalize.normalize(X)))


if __name__ == '__main__':
    main()
