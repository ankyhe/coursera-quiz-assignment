import numpy as np
import matplotlib.pyplot as plt

from utils import load_data

IMG_HEIGHT = 20


def main():
    data = load_data.load_mat('ex3data1.mat')
    X = data['X']
    sample = np.random.choice(X.shape[0], 100)

    for i in range(10):
        tmp = X[sample[10 * i : 10 * i + 10], :].reshape(-1, IMG_HEIGHT).T
        if i == 0:
            v = tmp
        else:
            v = np.vstack((v, tmp))

    plt.imshow(v)
    plt.axis('off');
    plt.show()


if __name__ == '__main__':
    main()