import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def load_image():
    data = plt.imread(r'data/bird_small.png')
    return data.reshape(-1, 3), data


def compress(img, original_image_shape, reduce_to_color):
    km = KMeans(reduce_to_color)
    km.fit(img)
    print('colors are {0}'.format(km.cluster_centers_))
    print(km.labels_)
    ret = km.cluster_centers_[km.labels_].reshape(original_image_shape[0], original_image_shape[1], 3)
    return ret


def main():
    img, original_image = load_image()
    compressed_img = compress(img, original_image.shape, 32)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 9))
    ax1.imshow(original_image)
    ax1.set_title('Original')
    ax2.imshow(compressed_img)
    ax2.set_title('Compressed, with 16 colors')
    for ax in fig.axes:
        ax.axis('off')
    plt.show()


if __name__ == '__main__':
    main()
