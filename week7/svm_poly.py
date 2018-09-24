import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

from utils import load_data


def load():
    data = load_data.load_mat('ex6data3.mat')
    X = data['X']
    y = data['y']
    return X, y


def plot(X, y, show = True):
    pos = (y == 1).ravel()
    neg = (y == 0).ravel()
    plt.scatter(X[pos, 0], X[pos, 1], s = 60, c = 'k', marker = '+', linewidths=1)
    plt.scatter(X[neg, 0], X[neg, 1], s = 60, c = 'y', marker = 'o', linewidths=1)
    if show:
        plt.show()


def plot_svc(svc, X, y, h=0.02, pad=0.25):
    x_min, x_max = X[:, 0].min()-pad, X[:, 0].max()+pad
    y_min, y_max = X[:, 1].min()-pad, X[:, 1].max()+pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.2)

    plot(X, y, show = False)
    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='r', marker='|', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)


def poly_svm(C, degree, gamma, X, y):
    svc = SVC(C = C, kernel = 'poly', degree = degree, gamma = gamma)
    svc.fit(X, y.ravel())
    plot_svc(svc, X, y)


def main():
    X, y = load()
    plot(X, y)
    poly_svm(1.0, 3, 10, X, y)


if __name__ == '__main__':
    main()