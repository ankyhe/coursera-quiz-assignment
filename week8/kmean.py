from functools import partial
import numpy as np

from utils import load_data, plot_data


def load():
    data = load_data.load_mat('ex7data2.mat')
    return data['X']


def plot(X, show = False):
    plot_data.plot(X[:, 0], X[:, 1], 'x1', 'x2',
        {
            'show': show
        })


def plot(X, idx_arr, centroids):
    colors = ['r', 'y', 'g', 'b']
    shapes = ['x', 'x', 'x', 'x']
    cluster_count = centroids.shape[0]
    for idx in range(cluster_count):
        row_idx_arr = np.argwhere(idx_arr == idx).ravel()
        X_idx = X[row_idx_arr]
        plot_data.plot(X_idx[:, 0], X_idx[:, 1], 'x1', 'x2',
                       {
                           'fmt': colors[idx] + shapes[idx],
                           'show': False
                       })
        plot_data.plot(centroids[idx, 0], centroids[idx, 1], 'x1', 'x2',
                    {
                        'fmt': 'k+',
                        'show': idx == cluster_count - 1
                    })


def random_init(X, cluster_count):
    row = X.shape[0]
    idx_arr = np.random.randint(row, size = cluster_count)
    centroids = X[idx_arr]
    return centroids


def find_closest_centroid(x, centroids):
    distance_power2_array = np.sum(np.power(x - centroids, 2), axis = 1)
    distance_power2_min_index = np.argmin(distance_power2_array)
    return distance_power2_min_index


def find_closest_for_all(X, centroids):
    find_closest_centroid_for_one = partial(find_closest_centroid, centroids = centroids)
    return np.apply_along_axis(find_closest_centroid_for_one, 1, X)


def compute_centroids(X, idx_arr, cluster_count, old_idx_arr, old_centroids):
    if np.array_equal(idx_arr, old_idx_arr):
        return old_centroids, False

    sums = np.zeros((cluster_count, X.shape[1]))
    counts = np.zeros(cluster_count)

    def add_to(x, idx):
        sums[idx, :] += x
        counts[idx] += 1

    for row_idx in range(X.shape[0]):
        x = X[row_idx, :]
        add_to(x, idx_arr[row_idx])

    non_zero_indexes = np.nonzero(counts)[0]
    sums = sums[non_zero_indexes]
    counts = counts[non_zero_indexes]
    return sums / counts.reshape(np.size(counts), -1), True


def k_mean_once(cluster_count):
    centroids = random_init(X, 3)
    old_idx_arr = None
    while True:
        idx_arr = find_closest_for_all(X, centroids)
        centroids, continued = compute_centroids(X, idx_arr, cluster_count, old_idx_arr, centroids)
        old_idx_arr = idx_arr
        if continued:
            continue
        break
    print('k_mean_once end')
    return centroids, idx_arr


def k_mean(cluster_count):
    min_distance = -1
    for idx in range(100):
        centroids, idx_arr = k_mean_once(cluster_count)
        distance = 0
        for idx in range(centroids.shape[0]):
            row_idx_arr = np.argwhere(idx_arr == idx).ravel()
            X_idx = X[row_idx_arr]
            distance += np.sum(np.power(X_idx - centroids[idx, :], 2)) / X_idx.shape[0]

        if min_distance == -1 or distance < min_distance:
            min_distance = distance
            ret_centroids = centroids
            ret_idx_arr = idx_arr

    return ret_centroids, ret_idx_arr




if __name__ == '__main__':
    X = load()
    #plot(X)

    cluster_count = 3  #3 is get from manual check with plot(X)
    centroids, idx_arr = k_mean(cluster_count)
    print(centroids)
    plot(X, idx_arr, centroids)






