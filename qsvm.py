from typing import Any, Callable
from itertools import product

import numpy as np
from numpy.typing import ArrayLike, NDArray
from dotenv import load_dotenv

from dwave.samplers import SimulatedAnnealingSampler

from kernels import _define_kernel

# Set the environment variable for your DWave API token
load_dotenv()


class QSVM:
    def __init__(
        self,
        B: int = 2,
        K: int = 3,
        zeta: float = 1.0,
        kernel: str | Callable = 'rbf',
        degree: int = 3,
        gamma: float = 1.0,
        coef0: float = 0.0,
    ) -> None:
        self.B = B
        self.K = K
        self.zeta = zeta
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0

        self._ud_ker = isinstance(kernel, Callable)
        self.kernel = kernel if self._ud_ker else _define_kernel(kernel, self.gamma, self.coef0, self.degree)

        # Maximum possible value of coefficients alpha_n
        self.C = (self.B ** np.arange(self.K)).sum()

        self.fitted: bool = False

        # Variables to be defined when the QSVM is fit
        self._computed_kernel: NDArray
        self._y_dtype: np.dtype
        self._y_label_map: dict[int, Any]
        self.X: NDArray
        self.y: NDArray
        self.n_train_samples: int
        self.qubo: NDArray
        self.alpha: NDArray
        self.bias: float

    def fit(self, X: ArrayLike, y: ArrayLike) -> None:
        '''Fit the QSVM on input data X and corresponding labels y

        Parameters
        ----------
        X : ArrayLike
            An array of shape (n_samples, n_features) of training data
        y : ArrayLike
            An array of shape (n_samples,) of class labels for the training data
        '''
        X, y = self._check_Xy(X, y)

        self.X = X
        self.y = y
        self.n_train_samples = X.shape[0]

        self._compute_kernel(X)
        self._compute_qubo(y)
        binary_vars = self._simulate_qubo()
        self._compute_coefficients(binary_vars)
        self._compute_bias()

        self.fitted = True

    def predict(self, X: ArrayLike):
        '''Perform classification on samples in X'''

        if not self.fitted:
            self._raise_not_fitted('predict')

        X = self._check_Xy(X)
        if X.shape[1] != self.X.shape[1]:
            raise ValueError(f'X has {X.shape[1]} features, but {self.X.shape[1]} features are expected as input')

        results = self.decision_function(X)

        preds_bin = np.sign(results).astype(np.int8)
        preds_bin[preds_bin == 0] = 1  # Go with class 1 if we don't know
        preds = np.empty(preds_bin.shape, dtype=self._y_dtype)
        for i in (-1, 1):
            preds[preds_bin == i] = self._y_label_map[i]
        return preds

    def decision_function(self, X: ArrayLike) -> NDArray:
        '''Evaluate the decision function on input samples X'''

        if not self.fitted:
            self._raise_not_fitted('decision_function')

        results = np.empty(X.shape[0])
        alpha_times_y = self.alpha * self.y
        for i, x in enumerate(X):  # Slow af
            results[i] = (alpha_times_y * self.kernel(self.X, x)).sum() + self.bias

        return results

    def _check_Xy(self, X: ArrayLike, y: ArrayLike | None = None) -> NDArray | tuple[NDArray, NDArray]:
        '''Check the passed in X and y data for validity and convert y to binary'''

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)
        if y is not None and not isinstance(y, np.ndarray):
            y = np.asarray(y)

        if len(X.shape) != 2:
            raise ValueError('X must be an array of shape (n_samples, n_features)')

        if y is None:
            return X

        if X.shape[0] != y.shape[0]:
            raise ValueError('X and y must have the same length')

        unique_y = np.unique(y)
        if len(unique_y) != 2:
            raise ValueError('Only binary classification is supported at this time')

        self._y_dtype = y.dtype
        self._y_label_map = {-1: unique_y[0], 1: unique_y[1]}

        y_bin = np.empty_like(y, np.int8)
        y_bin[y == self._y_label_map[-1]] = -1
        y_bin[y == self._y_label_map[1]] = 1

        return X, y_bin

    def _compute_kernel(self, X: NDArray) -> None:
        '''Compute the value of the kernel function on the input data'''

        self._computed_kernel = np.empty((self.n_train_samples, self.n_train_samples))
        if self._ud_ker:
            # If the kernel is user-defined, compute each element individually
            for i, j in product(range(self.n_train_samples), range(self.n_train_samples)):
                self._computed_kernel[i, j] = self.kernel(X[i], X[j])
        else:
            # Take advantage of the symmetry of the kernel function
            # Compute the upper triangle (including the diagonal)
            upper_triangle_indices = np.triu_indices(self.n_train_samples)
            result_upper_triangle = self.kernel(X[upper_triangle_indices[0]], X[upper_triangle_indices[1]])
            self._computed_kernel[upper_triangle_indices] = result_upper_triangle

            # Fill the lower triangle by copying from the upper triangle
            lower_triangle_indices = np.tril_indices(self.n_train_samples, -1)
            self._computed_kernel[lower_triangle_indices] = self._computed_kernel.T[lower_triangle_indices]

    def _compute_qubo(self, y: NDArray) -> None:
        '''Compute the QUBO matrix associated to the QSVM problem'''

        # Compute the coefficients without the base exponent or diagonal adjustments
        y_expanded = y[:, np.newaxis]
        coeff_matrix = 0.5 * y_expanded @ y_expanded.T * (self._computed_kernel + self.zeta)

        # Base exponent values
        k = np.arange(self.K).reshape(self.K, 1)
        j = np.arange(self.K)
        base_power = self.B ** (k + j)

        # Compute the complete matrix using the Kronecker product
        self.qubo = np.kron(coeff_matrix, base_power)

        # Adjust diagonal elements
        diag_indices = np.arange(self.K * self._computed_kernel.shape[0])
        self.qubo[diag_indices, diag_indices] -= np.tile(self.B**j, self._computed_kernel.shape[0])

        # Define the QUBO matrix, which is given by
        #     { Q_tilde[i, j] + Q_tilde[j, i]   if i < j
        # Q = { Q_tilde[i, j]                   if i == j
        #     { 0                               otherwise
        # self.qubo = np.triu(self.qubo) + np.tril(self.qubo, k=-1).T  # Cleaner, but slower
        lower_triangle_indices = np.tril_indices_from(self.qubo, -1)
        upper_triangle_indices = np.triu_indices_from(self.qubo, 1)
        self.qubo[lower_triangle_indices] = 0
        self.qubo[upper_triangle_indices] *= 2

    def _simulate_qubo(self) -> dict[int, int]:
        '''Run simulated annealing on the QUBO'''

        sampler = SimulatedAnnealingSampler()
        result = sampler.sample_qubo(self.qubo)
        return result.first.sample

    def _compute_coefficients(self, binary_vars: dict[int, int]) -> None:
        '''Compute the SVM coefficients from the binary coefficients returned by a QUBO sampler'''

        self.alpha = np.empty(self.n_train_samples, dtype=np.int32)
        base_powers = self.B ** np.arange(self.K)
        for n in range(self.n_train_samples):
            base_index = self.K * n
            self.alpha[n] = sum(base_powers[k] * binary_vars[base_index + k] for k in range(self.K))

        assert np.all(self.alpha <= self.C), 'A computed value of alpha is larger than possible'

    def _compute_bias(self) -> None:
        '''Compute the bias of the decision function'''

        denom = (self.alpha * (self.C - self.alpha)).sum()
        inner_sum = (self.alpha * self.y * self._computed_kernel).sum(axis=0)
        numer = (self.alpha * (self.C - self.alpha) * (self.y - inner_sum)).sum()
        self.bias = numer / denom
        # self.bias = 0  ###########################################################################################################

    @staticmethod
    def _raise_not_fitted(method: str):
        raise Exception(
            f'This QSVM instance is not fitted yet. Call \'fit\' with appropriate arguments before calling \'{method}\'.'
        )


qsvm = QSVM(kernel='rbf', B=2, K=3, gamma=0.25, zeta=1, )

np.random.seed(9485854)
x = 2 * np.random.rand(500, 2) - 1
y = np.empty(x.shape[0])
for i, xp in enumerate(x):
    y[i] = 1 if xp[0] ** 2 + xp[1] ** 2 < 0.75 else 0

qsvm.fit(x, y)

xt = 2 * np.random.rand(500, 2) - 1
preds = qsvm.predict(xt)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 8))
xtrain_in = x[y == 1][:, 0]
ytrain_in = x[y == 1][:, 1]
xtrain_out = x[y == 0][:, 0]
ytrain_out = x[y == 0][:, 1]
plt.scatter(xtrain_in, ytrain_in, c='b')
plt.scatter(xtrain_out, ytrain_out, c='r')
plt.show()

plt.figure(figsize=(8, 8))
xtrain_in = xt[preds == 1][:, 0]
ytrain_in = xt[preds == 1][:, 1]
xtrain_out = xt[preds == 0][:, 0]
ytrain_out = xt[preds == 0][:, 1]
plt.scatter(xtrain_in, ytrain_in, c='b')
plt.scatter(xtrain_out, ytrain_out, c='r')
plt.show()
