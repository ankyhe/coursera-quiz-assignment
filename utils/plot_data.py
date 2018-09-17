import matplotlib.pyplot as plt


def plot(X, Y, x_label, y_label, opts = None):
    opts = opts or {}
    fmt = opts.get('fmt', 'ro')
    title = opts.get('title', '')
    label = opts.get('label', '')
    legend_loc = opts.get('legend_loc')
    show = opts.get('show', False)
    markersize = opts.get('markersize', 10)

    plt.plot(X, Y, fmt, label = label, markersize = markersize)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if len(title) > 0:
        plt.title(title)
    if legend_loc:
        plt.legend(loc = legend_loc)
    if show:
        plt.draw()
        plt.show()
