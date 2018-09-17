import numpy as np

from utils import load_data, normal_eqn


def main():
    X, Y, Orignal_X = load_data.load_and_process('data2.txt')
    theta = normal_eqn.normal_eqn(X, Y)
    print(theta)

    house_size = 1650
    br_num = 3
    new_X = np.array([1, house_size, br_num])
    price = new_X.dot(theta).sum()
    print(price) # 293081.464334972 compare this with price in ex2_gradient_descent.py


if __name__ == '__main__':
    main()
