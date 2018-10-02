import warnings

import matplotlib.pyplot as plt
import numpy as np
from sklearn.covariance import EllipticEnvelope
import seaborn as sns

from utils import load_data, plot_data


warnings.simplefilter(action='ignore', category=FutureWarning)


def load(show = False):
    data = load_data.load_mat('ex8data1.mat')
    if show:
        print('X ================')
        print(data['X'])
        print('Xval ================')
        print(data['Xval'])
        print('yval ================')
        print(data['yval'])
    return data['X'], data['Xval'], data['yval']


def plot(X):
    plot_data.plot(X[:, 0], X[:, 1], 'x1', 'x2',
                   {
                       'markersize': 4,
                       'fmt': 'bx',
                       'show': True
                   })


def anomaly_detection(X):
    clf = EllipticEnvelope()
    clf.fit(X)
    y_pred = clf.decision_function(X).ravel()
    percentile = 1.9
    threshold = np.percentile(y_pred, percentile)
    print(threshold)
    outliers = y_pred < threshold

    xx, yy = np.meshgrid(np.linspace(0, 25, 200), np.linspace(0, 30, 200))
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    sns.distplot(y_pred, rug=True, ax=ax1)
    sns.distplot(y_pred[outliers], rug=True, hist=False, kde=False, norm_hist=True, color='r', ax=ax1)
    ax1.vlines(threshold, 0, 0.9, colors='r', linestyles='dotted',
               label='Threshold for {} percentile = {}'.format(percentile, np.round(threshold, 2)))
    ax1.set_title('Distribution of Elliptic Envelope decision function values')
    ax1.legend(loc='best')

    ax2.scatter(X[:, 0], X[:, 1], c='b', marker='x')
    ax2.scatter(X[outliers][:, 0], X[outliers][:, 1], c='r', marker='x', linewidths=2)
    ax2.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red', linestyles='dotted')
    ax2.set_title("Outlier detection")
    ax2.set_xlabel('Latency (ms)')
    ax2.set_ylabel('Throughput (mb/s)')

    plt.show()


def main():
    X, Xval, yval = load()
    #plot(X)
    anomaly_detection(X)


if __name__ == '__main__':
    main()