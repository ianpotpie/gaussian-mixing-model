import numpy as np
from numpy import sqrt, exp, pi, sum
from numpy.linalg import det, inv

eps = 1e-8  # this is added to denominators to prevent division by zero


def multivariate_normal(mean, cov, x):
    """
    Finds the probability density of a point "x", relative to a multivariate normal distribution
    that has mean "mean" and covariance matrix "cov".

    :param mean: a 1D array representing the mean point of the normal
    :param cov: a 2D array representing the covariance matrix of the normal
    :param x: a 1D representing point in the probability space
    :return: the probability density of x
    """
    return exp(-0.5 * (x - mean).T @ inv(cov) @ (x - mean)) / sqrt(det(2 * pi * cov))


class GaussianMixtureModel:
    def __init__(self, k):
        self.k = k  # the number of gaussian mixtures
        self.priors = None  # the priors gaussian distributions (the probability that a point comes from each gaussian)
        self.means = None  # the means of the gaussian distributions
        self.covars = None  # the covariance matrices of the gaussian distributions

    def train(self, X, n_epochs):
        """
        Runs Estimation-Maximization in order to optimize the parameters of the normals "n_epochs" times.
        It applies the following rules to the prior, mean, and variance.

        Prior_k = P(Z=k) = N_k / N_total

        Mean_k = P(Z=k | X)^T * x / N_k

        Covariance_k =  (x - Mean_k)^T * P(Z=k | X) * (x - Mean_k) / N_k

        :param n_epochs: the number of expectation-maximization steps to take
        :param X: a 2D array representing a list of points
        :return: None
        """
        n_examples, n_dims = X.shape[0], X.shape[1]
        self.priors = np.ones(self.k) / self.k  # initialize the priors as uniform distribution
        self.means = X[np.random.choice(n_examples, self.k, replace=False)]  # initialize the means to be points from X
        self.covars = np.stack([np.identity(n_dims) for _ in range(self.k)])  # initialize covariances to be uniform

        for _ in range(n_epochs):

            # expectation step
            prob_k_given_x = self.predict(X)  # the probabilities that each point (x) came from each distribution (k)

            # maximization step
            N = sum(prob_k_given_x, axis=0)  # the expected number of points originating from each distribution
            for k in range(self.k):
                N_k, mean_k = N[k], self.means[k]
                self.priors[k] = N_k / n_examples
                self.means[k] = prob_k_given_x[:, k] @ X / (N_k + eps)
                self.covars[k] = prob_k_given_x[:, k] * (X - mean_k).T @ (X - mean_k) / (N_k + eps)

    def predict(self, X):
        """
        Finds the conditional probabilities of each distribution given a point (x).
        This is the probability that each point (x) was produced by each probability distribution (k).
        The prediction simply applies bayes rule for each point-distribution pair.

        P(Z=k | X=x) = P(Z=k) * P(X=x | Z=k) / P(X=x)

        :param X: a 2D array where each row is a point for prediction
        :return: a 2D array where each row is a distribution over the mixtures
        """
        n_examples = X.shape[0]
        prob_k_given_x = np.zeros((n_examples, self.k))
        for i in range(n_examples):
            x, prob_x = X[i], 0
            for j in range(self.k):
                prob_k, mean_k, cov_k = self.priors[j], self.means[j], self.covars[j]
                prob_x_given_k = multivariate_normal(mean_k, cov_k, x)  # density of x given it is from distribution k
                prob_k_given_x[i][j] = prob_k * prob_x_given_k  # bayes rule, but missing the denominator P(X=x)
                prob_x += prob_k_given_x[i][j]

            prob_k_given_x[i] /= (prob_x + eps)  # here we complete bayes rule with P(X=x)

        return prob_k_given_x
