import sys
import numpy as np
from model import GaussianMixtureModel
import matplotlib.pyplot as plt

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'w', 'b']
maps = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
        'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
        'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
epochs = 50


def main():
    # import the data on which to run k gaussian mixtures
    data = []
    data_file = sys.argv[1]
    with open(data_file, mode='r') as f:
        for line in f:
            data.append([float(v) for v in line.strip().split()])
    X = np.array(data)

    # the number of gaussian mixtures to use
    k = int(sys.argv[2])

    # setup and train the model
    gmm = GaussianMixtureModel(k)
    gmm.train(X, epochs)

    # creates a mesh over the domain (projects dimensions >2 down to 2)
    x_domain = np.linspace(np.min(X[:, 0]) - 0.1, np.max(X[:, 0]) + 0.1, 200)
    y_domain = np.linspace(np.min(X[:, 1]) - 0.1, np.max(X[:, 1]) + 0.1, 200)
    x_mesh, y_mesh = np.meshgrid(x_domain, y_domain)
    n_dims = X.shape[1]
    points = np.pad(np.array([x_mesh.flatten(), y_mesh.flatten()]).T, [[0, 0], [0, n_dims - 2]])

    # draws the contours of the distributions
    soft_labels = gmm.predict(points).reshape(x_mesh.shape[0], x_mesh.shape[1], -1)
    for k in range(k):
        plt.contourf(x_mesh, y_mesh, soft_labels[:, :, k], cmap=maps[k + 1], alpha=1 / (k + 1))
    plt.scatter(X[:, 0], X[:, 1], facecolors="none", edgecolors="grey")
    for k in range(k):
        plt.scatter(gmm.means[k][0], gmm.means[k][1], color="black")
    plt.show()

    # draws the decision regions of the mixing space
    labels = np.argmax(soft_labels, axis=2)
    plt.pcolormesh(x_mesh, y_mesh, labels, shading="auto")
    plt.scatter(X[:, 0], X[:, 1], facecolors="none", edgecolors="grey")
    for k in range(k):
        plt.scatter(gmm.means[k][0], gmm.means[k][1], color="black")
    plt.show()


if __name__ == "__main__":
    main()
