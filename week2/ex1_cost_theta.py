import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from utils import load_data, computer_cost
import ex1_gradient_descent


def fill_theta_cost(X, Y):
    theta0_arr = np.linspace(-10, 10, 100)
    theta1_arr = np.linspace(-1, 4, 100)
    j_cost_arr = np.zeros((theta0_arr.size, theta1_arr.size))

    for i, theta0 in enumerate(theta0_arr):
        for j, theta1 in enumerate(theta1_arr):
            theta = np.array([[theta0], [theta1]])
            j_cost = computer_cost.computer(X, Y, theta)
            j_cost_arr[i, j] = j_cost
    return theta0_arr, theta1_arr, j_cost_arr


def main():
    X, Y, _ = load_data.load_and_process('data1.txt')

    theta0_arr, theta1_arr, j_cost_arr = fill_theta_cost(X, Y)

    j_cost_arr = j_cost_arr.T

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    theta0_arr, theta1_arr = np.meshgrid(theta0_arr, theta1_arr)
    surf = ax.plot_surface(theta0_arr, theta1_arr, j_cost_arr, cmap = cm.coolwarm, rstride = 2, cstride = 2)
    fig.colorbar(surf)
    plt.xlabel('theta0')
    plt.ylabel('theta1')
    plt.show()

    alpha = 0.023
    iterations = 400
    fig = plt.figure()
    fig.add_subplot(111)
    cset = plt.contour(theta0_arr, theta1_arr, j_cost_arr, np.logspace(-2, 3, 20), cmap = cm.coolwarm)
    fig.colorbar(cset)
    plt.xlabel('theta0')
    plt.ylabel('theta1')

    theta, _ = ex1_gradient_descent.gradient_descent(X, Y, alpha, iterations)
    plt.plot(theta[0, 0], theta[1, 0], 'rx', markersize=10, linewidth=2)
    plt.show()


if __name__ == '__main__':
    main()
