from utils import load_data, normal_eqn


def main():
    X, Y, Orignal_X = load_data.load_and_process('data2.txt')
    theta = normal_eqn.normal_eqn(X, Y)
    print(theta)


if __name__ == '__main__':
    main()
