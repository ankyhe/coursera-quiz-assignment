import numpy as np

from utils import sigmoid, plot_data


def main():
    x = np.linspace(-8, 8, 1000)
    y = sigmoid.sigmoid(x)
    plot_data.plot(x, y, 'x', 'y',
                   {
                       'fmt': 'b-',
                       'title': 'sigmoid',
                       'label': 'sigmoid',
                       'show': False
                   })
    y2 = []
    for _ in x:
        y2.append(0.5)
    y2 = np.array(y2)
    plot_data.plot(x, y2, 'x', 'y',
                   {
                       'fmt': 'g-',
                       'label': 'y = 0.5',
                       'show': True
                   })


if __name__ == '__main__':
    main()