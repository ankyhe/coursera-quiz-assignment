import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from scipy import linalg


from utils import load_data, plot_data


def load():
    data = load_data.load_mat('ex7data1.mat')
    return data['X']


def plot(X, show = False):
    plot_data.plot(X[:, 0], X[:, 1], 'x1', 'x2',
        {
            'markersize': 5,
            'fmt': 'kx',
            'show': show
        })


def computer_svd(X):
    standard_scaler = StandardScaler()
    standard_scaler.fit(X)
    return linalg.svd(standard_scaler.transform(X).T) + (standard_scaler, )


def main():
    X = load()
    #plot(X, False)
    U, S, V, scaler = computer_svd(X)
    plt.scatter(X[:,0], X[:,1], s=30, edgecolors='b',facecolors='None', linewidth=1)
    plt.gca().set_aspect('equal')
    plt.quiver(scaler.mean_[0], scaler.mean_[1], U[0,0], U[0,1], scale=S[1], color='r')
    plt.quiver(scaler.mean_[0], scaler.mean_[1], U[1,0], U[1,1], scale=S[0], color='y')
    plt.show()


if __name__ == '__main__':
    main()
