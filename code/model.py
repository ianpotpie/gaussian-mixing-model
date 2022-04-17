import numpy as np
from numpy import sqrt, exp, pi, sum
from numpy.linalg import det, inv

eps = 1e-8


def multivariate_normal(mean, cov, x):
    """
    Finds the probability density function of a point "x", relative to a multivariate normal distribution
    that has mean "mean" and covariance matrix "cov".

    :param mean: a 1D array representing the mean point of the normal
    :param cov: a 2D array representing the covariance matrix of the normal
    :param x: a 1D representing point in the probability space
    :return: the probability density of all point in X
    """
    return exp(-0.5 * (x - mean).T @ inv(cov) @ (x - mean)) / sqrt(det(2 * pi * cov))


class GaussianMixtureModel:
    def __init__(self, k):
        self.k = k  # the number of gaussian mixtures
        self.priors = None
        self.means = None
        self.covars = None

    def train(self, X, n_epochs):
        """
        Runs Estimation-Maximization in order to optimize the parameters of the normals.

        :param n_epochs: the number of expectation-maximization steps to take
        :param X: a 2D array representing a list of points
        :return: None
        """

        # initialization
        n_examples, n_dims = X.shape[0], X.shape[1]
        self.priors = np.ones(self.k) / self.k
        self.means = X[np.random.choice(n_examples, self.k, replace=False)]
        self.covars = np.stack([np.identity(n_dims) for _ in range(self.k)])
        prob_y_given_x = np.zeros((n_examples, self.k))

        for _ in range(n_epochs):

            # expectation step
            for i in range(n_examples):
                x, prob_x = X[i], 0
                for j in range(self.k):
                    prob_y, mean, cov = self.priors[j], self.means[j], self.covars[j]
                    prob_x_given_y = multivariate_normal(mean, cov, x)
                    prob_y_given_x[i][j] = prob_y * prob_x_given_y  # notice the P(X=x) is missing from bayes rule
                    prob_x += prob_y_given_x[i][j]

                prob_y_given_x[i] /= (prob_x + eps)  # here we complete bayes rule with P(X=x)

            # maximization step
            y_expectations = sum(prob_y_given_x, axis=0)
            for j in range(self.k):
                y_expectation, mean = y_expectations[j], self.means[j]
                self.priors[j] = y_expectation / n_examples
                self.means[j] = prob_y_given_x[:, j] @ X / (y_expectation + eps)
                self.covars[j] = prob_y_given_x[:, j] * (X - mean).T @ (X - mean) / (y_expectation + eps)

    def predict(self, X):
        """
        Finds the probability of some points across the mixture of distributions.

        :param X: a 2D array where each row is a point for prediction
        :return: a 2D array where each row is a distribution over the mixtures
        """
        n_examples = X.shape[0]
        prob_y_given_x = np.zeros((n_examples, self.k))
        for i in range(n_examples):
            x, prob_x = X[i], 0
            for j in range(self.k):
                prob_y, mean, cov = self.priors[j], self.means[j], self.covars[j]
                prob_x_given_y = multivariate_normal(mean, cov, x)
                prob_y_given_x[i][j] = prob_y * prob_x_given_y  # notice the P(X=x) is missing from bayes rule
                prob_x += prob_y_given_x[i][j]

            prob_y_given_x[i] /= (prob_x + eps)  # here we complete bayes rule with P(X=x)

        return prob_y_given_x

