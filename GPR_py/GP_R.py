from scipy.spatial.distance import pdist, cdist, squareform
import numpy as np
from numpy.linalg import cholesky
from scipy.optimize import minimize
from numpy.linalg import inv

class GPR():
    def __init__(self):
        self.kernel = self.Matern


    def Matern(self, X1, X2, l=1.0):
        dists = cdist(X1, X2, metric='sqeuclidean')
        m = 1 + np.sqrt(5) * np.sqrt(dists) / l + 5 / 3 * dists / np.square(l)
        n = np.exp(-np.sqrt(5) * np.sqrt(dists) / l)
        K = m * n
        return K

    def nll_fn(self, theta, X_train, Y_train):
        K = self.kernel(X_train, X_train, l=theta[0]) + theta[1] ** 2 * np.eye(len(X_train))
        # Compute determinant via Cholesky decomposition
        return np.sum(np.log(np.diagonal(cholesky(K)))) + \
               0.5 * Y_train.T.dot(inv(K).dot(Y_train)) + \
               0.5 * len(X_train) * np.log(2 * np.pi)

    def grad(self, theta, X_train, Y_train):
        K = self.kernel(X_train, X_train, l=theta[0]) + theta[1] ** 2 * np.eye(len(X_train))
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train)).ravel()
        dists = squareform(pdist(X_train, 'sqeuclidean'))
        m = 1 + np.sqrt(5) * np.sqrt(dists) / theta[0] + 5 / 3 * dists / np.square(theta[0])
        n = np.exp(-np.sqrt(5) * np.sqrt(dists) / theta[0])

        grad_1 = -(np.sqrt(5) * np.sqrt(dists) / np.square(theta[0]) + 10 * dists / (
                3 * np.square(theta[0]) * theta[0])) * n
        grad_2 = n * (np.sqrt(5) * np.sqrt(dists) / np.square(theta[0])) * m
        l_gra = grad_1 + grad_2
        sigma_gra = 2 * theta[1] * np.eye(len(X_train))

        tmp = np.einsum("i,j->ij", alpha, alpha)

        tmp -= np.linalg.solve(L.T, np.linalg.solve(L, np.eye(K.shape[0])))  # K inverse
        log_likelihood_gradient_l = -0.5 * np.einsum("ij,ijk->k", tmp, l_gra[:, :, np.newaxis])[0]
        log_likelihood_gradient_sigma = -0.5 * np.einsum("ij,ijk->k", tmp, sigma_gra[:, :, np.newaxis])[0]
        gradient = np.array([log_likelihood_gradient_l, log_likelihood_gradient_sigma])
        return gradient

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        res = minimize(self.nll_fn, [1, 1e-10], args=(X, y), jac=self.grad,
                       bounds=((1e-5, None), (1e-5, None)),
                       method='L-BFGS-B')

        self.l = res['x'][0]
        self.sigma_y = res['x'][1]
        print('Optimized l and sigma_y: ',(self.l,self.sigma_y))

    def predict(self, X_test):
        K = self.kernel(self.X_train, self.X_train, self.l) + self.sigma_y ** 2 * np.eye(len(self.X_train))
        K_s = self.kernel(self.X_train, X_test, self.l)
        K_ss = self.kernel(X_test, X_test, self.l) + 1e-8 * np.eye(len(X_test))
        K_inv = inv(K)

        mu_s = K_s.T.dot(K_inv).dot(self.y_train)

        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)

        return mu_s, cov_s