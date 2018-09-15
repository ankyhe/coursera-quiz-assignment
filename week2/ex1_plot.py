from utils import load_data
from utils import plot_data


def main():
    data = load_data.load('data1.txt')
    X = data[:, 0]
    Y = data[:, 1]
    plot_data.plot(X, Y, 'house size', 'house price',
                   {
                       'show': True,
                       'title': 'Original Data',
                       'fmt': 'rx'
                   })


if __name__ == '__main__':
    main()
