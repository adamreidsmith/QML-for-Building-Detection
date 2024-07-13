import warnings
import time
from typing import Optional, Any
from collections.abc import Callable
from functools import partial

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler, SteepestDescentSampler
from dwave.system import LeapHybridSampler, DWaveSampler, AutoEmbeddingComposite
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel


def retry(
    func: Callable[..., Any],
    max_retries: int = 2,
    exception: Exception = Exception,
    delay: float = 0,
    warn: bool = False,
    *args: Any,
    **kwargs: Any,
) -> Any:
    '''
    Attempts to call the provided function with specified arguments and keyword arguments.
    Retries the function up to a maximum number of retries if an exception occurs.

    Parameters
    ----------
    func : Callable[..., Any]
        The function to be called.
    max_retries : int
        The maximum number of attempts to call the function. Default is 2.
    exception : Exception
        Retry only if this exception is raised. Default is Exception.
    delay : float
        Delay for this many seconds before retrying. Default is 0.
    warn : bool
        Whether or not to warn when the function call needs to be retried. Default is False.
    *args
        Variable length argument list to pass to the function.
    **kwargs
        Arbitrary keyword arguments to pass to the function.

    Returns
    -------
    Any
        The return value of the function if it succeeds.

    Raises
    ------
    Exception
        Re-raises the exception from the function if the maximum number of retries is reached.

    Warnings
    --------
    RuntimeWarning
        Issues a warning if a retry attempt fails.
    '''

    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except exception as e:
            if attempt == max_retries - 1:
                raise e from None
            if warn:
                warn_msg = (
                    f'Attempt {attempt + 1}/{max_retries} failed for function \'{func.__name__}\'. '
                    f'Error: {type(e).__name__}: {str(e)}. '
                    f'Retrying in {delay} seconds...'
                )
                warnings.warn(warn_msg, RuntimeWarning)
            if delay > 0:
                time.sleep(delay)


class QSVM:
    '''Quantum support vector machine'''

    valid_kernels = {'linear', 'rbf', 'sigmoid', 'poly'}
    valid_samplers = {'simulate', 'steepest_descent', 'tabu', 'hybrid', 'qa'}

    _dwave_sampler: Optional[DWaveSampler] = None
    _hybrid_sampler: Optional[LeapHybridSampler] = None

    def __init__(
        self,
        kernel: str = 'rbf',
        B: int = 2,
        P: int = 0,
        K: int = 3,
        zeta: float = 1.0,
        gamma: float = 1.0,
        coef0: float = 0.0,
        degree: int = 3,
        sampler: str = 'steepest_descent',
        num_reads: int = 100,
        hybrid_time_limit: float = 3,
        normalize: bool = True,
        warn: bool = False,
    ) -> None:
        '''
        Parameters
        ----------
        kernel : str
            Specifies the kernel type to be used in the algorithm. One of 'linear', 'rbf', 'poly', or 'sigmiod'.
            Default is 'rbf'.
        B : int
            Base to use in the binary encoding of the QSVM coefficients. See equation (10) in [1]. Default is 2.
        P : int
            The shift parameter in the encoding exponent that allows for negative exponents. Default is 0.
        K : int
            The number of binary encoding variables to use in the encoding of the QSVM coefficients. See equation
            (10) in [1]. Default is 3.
        zeta : float
            A parameter of the QSVM which enforces one of the constraints of a support vector machine in the QSVM
            QUBO model. See equation (11) in [1]. Default is 1.
        gamma : float
            Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. Must be non-negative. Default is 1.
        coef0 : float
            Independent term in kernel function. It is only significant in 'poly' and 'sigmoid' kernels.
            Default is 0.
        degree : int
            Degree of the polynomial kernel function ('poly'). Must be non-negative. Ignored by all other kernels.
            Default is 3.
        sampler : str
            The sampler used for annealing. One of 'qa', 'simulate', 'steepest_descent', 'tabu', or 'hybrid'. Only 'qa'
            and 'hybrid' use real quantum hardware. 'qa' will fail if the problem is too large to easily embed on the
            quantum annealer.  Default is 'steepest_descent'.
        num_reads : int
            Number of reads of the quantum or simulated annealer to compute the QSVM solution. Default is 100.
        hybrid_time_limit : float
            The time limit in seconds for the hybrid solver. Default is 3.
        normalize : bool
            Whether or not to normalize input data. Default is True.
        warn : bool
            Warn if samples lie on the decision boundary of the fitted classifier. Default is False.

        References
        ----------
        [1] Willsch, Willsch, De Raedt, and Michielsen, 2020. "Support vector machines on the D-Wave quantum annealer".
            https://www.sciencedirect.com/science/article/pii/S001046551930342X. DOI: 10.1016/j.cpc.2019.107006
        '''

        self.B = float(B)  # Base of the encoding
        self.P = P  # Encoding exponent shift
        self.K = K  # Number of encoding variables
        self.zeta = zeta  # Constraint term coefficient

        # Kernel parameters
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree

        # Annealing method parameters
        if sampler not in self.valid_samplers:
            raise ValueError(f'Invalid sampler \'{sampler}\'. Valid options are {self.valid_samplers}')
        self.sampler = sampler
        self.num_reads = num_reads

        if kernel not in self.valid_kernels:
            raise ValueError(f'Invalid kernel \'{kernel}\'. Valid options are {self.valid_kernels}')
        self.kernel = self._define_kernel(kernel, self.gamma, self.coef0, self.degree)

        self.normalize = normalize
        self.hybrid_time_limit = hybrid_time_limit

        self.warn = warn

        # Types for various instance variables that are exposed during training
        self.qubo: dict[tuple[int, int], float] | None = None
        self.N: int
        self.X: np.ndarray
        self.y: np.ndarray
        self._computed_kernel: np.ndarray
        self._normalization_mean: np.ndarray
        self._normalization_stdev: np.ndarray
        self.alphas: np.ndarray
        self.bias: float

    def _define_kernel(
        self, ker: str, gamma: Optional[float] = None, coef0: Optional[float] = None, degree: Optional[int] = None
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        '''
        Define the kernel function.
        '''

        if ker == 'linear':
            kernel = linear_kernel
        elif ker == 'rbf':
            kernel = partial(rbf_kernel, gamma=gamma)
        elif ker == 'sigmoid':
            kernel = partial(sigmoid_kernel, gamma=gamma, coef0=coef0)
        elif ker == 'poly':
            kernel = partial(polynomial_kernel, degree=degree, gamma=gamma, coef0=coef0)
        else:
            raise ValueError(f'Unknown kernel: {ker}')
        return kernel

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QSVM':
        '''
        Fit the QSVM on input data X and corresponding labels y.

        Parameters
        ----------
        X : np.ndarray
            An array of shape (n_samples, n_features) of training data.
        y : np.ndarray
            An array of shape (n_samples,) of binary class labels for the training data.
        '''

        # Ensure the data is formatted correctly
        assert (
            isinstance(X, np.ndarray) and X.ndim == 2
        ), 'X must be a 2-dimensional numpy array of shape (n_samples, n_features)'
        assert isinstance(y, np.ndarray) and y.ndim == 1, 'y must be a 1-dimensional numpy array of shape (n_samples,)'
        assert len(X) == len(y), 'X and y must be the same length'
        assert np.all(np.isin(y, (-1, 1))), 'y must contain only ±1'

        self.N = len(X)  # Number of train points
        self.X = self._normalize(X, train=True)  # Normalize the data
        self.y = y.astype(np.int8)

        # Compute the kernel matrix from the train data
        self._computed_kernel = self.kernel(self.X, self.X)

        # self.qubo = self._compute_qubo(self.y)
        self.qubo = self._compute_qubo_fast(self.y)

        sample = self._run_sampler()
        self.alphas = self._compute_qsvm_coefficients(sample)
        self.bias = self._compute_bias()
        return self

    def _normalize(self, X: np.ndarray, train: bool = False) -> np.ndarray:
        '''
        Normalize the input X w.r.t. the train data. If train=True, the mean and standard deviation are computed from X.
        '''

        if not self.normalize:
            return X
        if train:
            self._normalization_mean = np.mean(X, axis=0)
            self._normalization_stdev = np.std(X, axis=0)
        return (X - self._normalization_mean) / self._normalization_stdev

    def _compute_qubo(self, y: np.ndarray) -> np.ndarray:
        '''
        Compute the QUBO matrix associated to the QSVM problem from the precomputed kernel
        and the targets y.
        '''

        coeff_matrix = self._computed_kernel + self.zeta  # N x N array
        for n in range(self.N):
            for m in range(self.N):
                coeff_matrix[n, m] *= 0.5 * y[n] * y[m]

        base_power = np.empty((self.K, self.K))  # K x K array
        for k in range(self.K):
            for j in range(self.K):
                base_power[k, j] = self.B ** (k + j - 2 * self.P)

        # Formulate the QUBO as an upper triangular matrix
        qubo = {}
        for n in range(self.N):
            for m in range(self.N):
                for k in range(self.K):
                    for j in range(self.K):
                        kk = self.K * n + k
                        jj = self.K * m + j
                        if kk < jj:
                            qubo[(kk, jj)] = 2 * coeff_matrix[n, m] * base_power[k, j]
                        elif kk == jj:
                            qubo[(kk, jj)] = coeff_matrix[n, m] * base_power[k, j] - self.B ** (k - self.P)
        return qubo

    def _compute_qubo_fast(self, y: np.ndarray) -> np.ndarray:
        '''
        Compute the QUBO matrix associated to the QSVM problem from the precomputed kernel
        and the targets y. This performs the same operation as self._compute_qubo but utilizes
        numpy array methods for parallelization.
        '''

        # Compute the coefficients without the base exponent or diagonal adjustments
        y_expanded = y[:, np.newaxis]
        coeff_matrix = 0.5 * y_expanded @ y_expanded.T * (self._computed_kernel + self.zeta)

        # Base exponent values
        k = np.arange(self.K).reshape(self.K, 1) - self.P
        j = np.arange(self.K) - self.P
        base_power = self.B ** (k + j)

        # Compute the complete matrix using the Kronecker product
        qubo_array = np.kron(coeff_matrix, base_power)

        # Adjust diagonal elements
        diag_indices = np.arange(self.K * self._computed_kernel.shape[0])
        qubo_array[diag_indices, diag_indices] -= np.tile(self.B**j, self._computed_kernel.shape[0])

        # Define the QUBO matrix, which is given by
        #     { Q_tilde[i, j] + Q_tilde[j, i]   if i < j
        # Q = { Q_tilde[i, j]                   if i == j
        #     { 0                               otherwise
        lower_triangle_indices = np.tril_indices_from(qubo_array, -1)
        upper_triangle_indices = np.triu_indices_from(qubo_array, 1)
        qubo_array[lower_triangle_indices] = 0
        qubo_array[upper_triangle_indices] *= 2

        qubo = {(i, j): qubo_array[(i, j)] for i in range(self.K * self.N) for j in range(i, self.K * self.N)}
        return qubo

    def _run_sampler(self) -> dict[int, int]:
        '''
        Run the hybrid sampler or the simulated annealing sampler. Returns a dict that maps the
        indices of the binary variables to their values.
        '''

        if self.sampler == 'hybrid':
            return retry(self._run_hybrid_sampler, max_retries=2, delay=1, warn=True)
        return retry(self._run_pure_sampler, max_retries=2, delay=1, warn=True)

    def _run_pure_sampler(self) -> dict[int, int]:
        '''
        Run a purely quantum or classical sampling method on self.qubo.
        '''

        if self.sampler == 'simulate':
            sampler = SimulatedAnnealingSampler()
        elif self.sampler == 'tabu':
            sampler = TabuSampler()
        elif self.sampler == 'steepest_descent':
            sampler = SteepestDescentSampler()
        elif self.sampler == 'qa':
            if self._dwave_sampler is None:
                self._set_dwave_sampler()
            sampler = AutoEmbeddingComposite(self._dwave_sampler)
        else:
            raise ValueError(f'Unknown pure sampler: {self.sampler}')

        sample_set = sampler.sample_qubo(self.qubo, num_reads=self.num_reads)
        return sample_set.first.sample

    def _run_hybrid_sampler(self) -> dict[int, int]:
        '''
        Run the Leap hybrid sampler on self.qubo.
        '''

        if self._hybrid_sampler is None:
            self._set_hybrid_sampler()
        sample_set = self._hybrid_sampler.sample_qubo(self.qubo, time_limit=self.hybrid_time_limit)
        return sample_set.first.sample

    @classmethod
    def _set_hybrid_sampler(cls):
        '''
        Set the hybrid sampler as a class attribute so we can reuse the instance over multiple QSVM instances.
        This prevents the creation of too many threads when many QSVM instances are initialized.
        '''

        cls._hybrid_sampler = LeapHybridSampler()

    @classmethod
    def _set_dwave_sampler(cls):
        '''
        Set the DWave sampler as a class attribute so we can reuse the instance over multiple QSVM instances.
        This prevents the creation of too many threads when many QSVM instances are initialized.
        '''

        cls._dwave_sampler = DWaveSampler()

    def _compute_qsvm_coefficients(self, sample: dict[int, int]) -> np.ndarray:
        '''
        Compute the coefficients of the QSVM from the annealing solution. Decodes the binary variable encoding.
        '''

        alphas = np.empty(self.N)
        powers_of_b = self.B ** (np.arange(self.K) - self.P)
        for n in range(self.N):
            alphas[n] = sum(powers_of_b[k] * sample[self.K * n + k] for k in range(self.K))
        return alphas

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        '''
        Evaluate the decision function on input samples X.

        Parameters
        ----------
        X : np.ndarray
            An array of shape (n_samples, n_features) on which to evaluate the decision function.

        Returns
        -------
        np.ndarray
            The unthresholded results of the QSVM.
        '''

        if self.qubo is None:
            raise RuntimeError(
                f'This QSVM instance is not fitted yet. Call \'fit\' with appropriate arguments before calling \'decision_function\''
            )
        return self._decision_function_no_bias(X) + self.bias

    def _decision_function_no_bias(self, X: np.ndarray) -> np.ndarray:
        '''
        The decision function without the bias term. This is implemented separately from self.decision_function
        for use with bias optimization.
        '''

        # kernel_vals = self.kernel(self.X, X)
        # return sum(self.alphas[n] * self.y[n] * kernel_vals[n, :] for n in range(self.N))
        return (self.alphas * self.y) @ self.kernel(self.X, X)  # Same as above 2 lines, but faster

    def _compute_bias(self, n_biases: int = 100) -> float:
        '''
        Compute the bias which best fits the training data.
        This checks for a bias in the range [-results.max(), -results.min()] using `n_biases` tests
        and returns the one which maximizes classification accuracy of the training data.
        '''

        results = self._decision_function_no_bias(self.X)

        def compute_acc(bias: float) -> float:
            preds = results + bias
            preds = np.sign(preds).astype(np.int8)
            acc = (preds == self.y).sum() / len(self.y)
            return acc

        biases = np.linspace(-results.max(), -results.min(), n_biases)  # Biases to test
        accuracies = np.fromiter(map(compute_acc, biases), dtype=np.float64)  # Compute the accuracies for each bias
        best_acc_indices = (accuracies == accuracies.max()).nonzero()[0]  # Find indices of biases with best accuracy
        best_acc_index = best_acc_indices[len(best_acc_indices) // 2]  # Choose the middle bias index
        return biases[best_acc_index]

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Perform classification on samples in X.

        Parameters
        ----------
        X : np.ndarray
            An array of shape (n_samples, n_features) to classify.

        Returns
        -------
        np.ndarray
            The classification results.
        '''

        if self.qubo is None:
            raise RuntimeError(
                f'This QSVM instance is not fitted yet. Call \'fit\' with appropriate arguments before calling \'predict\''
            )

        if X.shape[1] != self.X.shape[1]:
            raise ValueError(f'X has {X.shape[1]} features, but {self.X.shape[1]} features are expected as input')

        # Normalize the input data w.r.t. the training data
        X = self._normalize(X)

        # Compute the predictions
        results = self.decision_function(X)
        preds = np.sign(results).astype(np.int8)

        # Set unclassified samples arbitrarily
        if np.any(preds == 0):
            if self.warn:
                warnings.warn(
                    f'{sum(preds == 0)} samples lie on the decision boundary. Arbitrarily assigning class 1.'
                )
            preds[preds == 0] = 1  # Go with class 1 if we don't know
        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        '''
        Computes the accuracy of the QSVM on inputs X against targets y.

        Parameters
        ----------
        X : np.ndarray
            The input data on which to run the QSVM.
        y : np.ndarray
            The target classes.

        Returns
        -------
        float
            The accuracy of the QSVM.
        '''

        y = y.astype(np.int8)

        preds = self.predict(X)  # Normalization happens here
        acc = (preds == y).sum() / len(y)
        return acc

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
