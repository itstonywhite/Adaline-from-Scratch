
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    Plot the decision regions of a classifier for 2D data.

    Parameters
    ----------
    X : array-like, shape (n_examples, 2)
        Feature matrix.
    y : array-like, shape (n_examples,)
        Class labels. At most 5 distinct classes.
    classifier : object
        Fitted classifier with a predict() method.
    resolution : float, default=0.02
        Grid step size in feature units.
    """

    X = np.asarray(X)
    y = np.asarray(y)

    if X.ndim != 2 or X.shape[1] != 2:
        raise ValueError("X must contain exactly two features.")

    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples.")

    if resolution <= 0:
        raise ValueError("resolution must be positive.")

    # Setup markers and color map
    markers = ('o', 's', '^', 'v', '<')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')

    classes = np.unique(y)

    if len(classes) > len(markers):
        raise ValueError("A maximum of 5 classes is supported.")

    cmap = ListedColormap(colors[:len(classes)])

    # Plot decision surface
    x1_min = X[:, 0].min() - 1
    x1_max = X[:, 0].max() + 1

    x2_min = X[:, 1].min() - 1
    x2_max = X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    grid = np.column_stack((xx1.ravel(), xx2.ravel()))

    lab = classifier.predict(grid)
    lab = lab.reshape(xx1.shape)

    plt.contourf(
        xx1,
        xx2,
        lab,
        alpha=0.3,
        cmap=cmap
    )

    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # Plot class labels
    for idx, cl in enumerate(classes):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=f"Class {cl}",
            edgecolors="black"
        )

# Tony White ✍️
