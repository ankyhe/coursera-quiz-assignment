from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(6)


def map(x):
    return poly.fit_transform(x)

