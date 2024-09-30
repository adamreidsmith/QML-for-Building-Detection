import warnings
import time
from typing import Optional, Any
from collections.abc import Callable
from functools import partial

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler, SteepestDescentSampler
from dwave.system import LeapHybridSampler, DWaveSampler, AutoEmbeddingComposite
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel
from scipy.special import softmax

from concurrent.futures import ThreadPoolExecutor


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
    '''
    Quantum Support Vector Machine

    References
    ----------
    [1] D. Willsch, M. Willsch, H. De Raedt, and K. Michielsen. "Support vector machines on the d-wave quantum
        annealer". Computer Physics Communications, 248:107006, 2020.
        https://www.sciencedirect.com/science/article/pii/S001046551930342X.
    '''

    valid_kernels = {'linear', 'rbf', 'sigmoid', 'poly'}
    valid_samplers = {'simulate', 'steepest_descent', 'tabu', 'hybrid', 'qa'}

    _dwave_sampler: Optional[DWaveSampler] = None
    _hybrid_sampler: Optional[LeapHybridSampler] = None

    _arg_defaults = {
        'kernel': 'rbf',
        'B': 2,
        'P': 0,
        'K': 3,
        'zeta': 1.0,
        'gamma': 1.0,
        'coef0': 0.0,
        'degree': 3,
        'sampler': 'steepest_descent',
        'num_reads': 100,
        'hybrid_time_limit': 3,
        'normalize': True,
        'warn': False,
    }

    def __init__(
        self,
        kernel: str | Callable[[np.ndarray, np.ndarray], float] = _arg_defaults['kernel'],
        B: int = _arg_defaults['B'],
        P: int = _arg_defaults['P'],
        K: int = _arg_defaults['K'],
        zeta: float = _arg_defaults['zeta'],
        gamma: float = _arg_defaults['gamma'],
        coef0: float = _arg_defaults['coef0'],
        degree: int = _arg_defaults['degree'],
        sampler: str = _arg_defaults['sampler'],
        num_reads: int = _arg_defaults['num_reads'],
        hybrid_time_limit: float = _arg_defaults['hybrid_time_limit'],
        normalize: bool = _arg_defaults['normalize'],
        warn: bool = _arg_defaults['warn'],
    ) -> None:
        f'''
        Parameters
        ----------
        kernel : str | Callable[[np.ndarray, np.ndarray], float]
            Specifies the kernel type to be used in the algorithm. If a string, one of 'linear', 'rbf', 'poly',
            or 'sigmiod'. If a Callable, the kernel elements K[i, j] will be computed as kernel(X[i], Y[j]).
            Default is {self._arg_defaults['kernel']}.
        B : int
            Base to use in the binary encoding of the QSVM coefficients. See equation (10) in [1].
            Default is {self._arg_defaults['B']}.
        P : int
            The shift parameter in the encoding exponent that allows for negative exponents.
            Default is {self._arg_defaults['P']}.
        K : int
            The number of binary encoding variables to use in the encoding of the QSVM coefficients. See equation
            (10) in [1]. Default is {self._arg_defaults['K']}.
        zeta : float
            A parameter of the QSVM which enforces one of the constraints of a support vector machine in the QSVM
            QUBO model. See equation (11) in [1]. Default is {self._arg_defaults['zeta']}.
        gamma : float
            Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. Must be non-negative.
            Default is {self._arg_defaults['gamma']}.
        coef0 : float
            Independent term in kernel function. It is only significant in 'poly' and 'sigmoid' kernels.
            Default is {self._arg_defaults['coef0']}.
        degree : int
            Degree of the polynomial kernel function ('poly'). Must be non-negative. Ignored by all other kernels.
            Default is {self._arg_defaults['degree']}.
        sampler : str
            The sampler used for annealing. One of 'qa', 'simulate', 'steepest_descent', 'tabu', or 'hybrid'. Only 'qa'
            and 'hybrid' use real quantum hardware. 'qa' will fail if the problem is too large to easily embed on the
            quantum annealer.  Default is {self._arg_defaults['sampler']}.
        num_reads : int
            Number of reads of the quantum or simulated annealer to compute the QSVM solution.
            Default is {self._arg_defaults['num_reads']}.
        hybrid_time_limit : float
            The time limit in seconds for the hybrid solver. Default is {self._arg_defaults['hybrid_time_limit']}.
        normalize : bool
            Whether or not to normalize input data. Default is {self._arg_defaults['normalize']}.
        warn : bool
            Warn if samples lie on the decision boundary of the fitted classifier.
            Default is {self._arg_defaults['warn']}.
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

        if kernel not in self.valid_kernels and not callable(kernel):
            raise ValueError(f'Invalid kernel \'{kernel}\'. Valid options are {self.valid_kernels} or a callable')
        self.kernel = kernel
        self.kernel_func = self._define_kernel(kernel, self.gamma, self.coef0, self.degree)

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
        self,
        ker: str | Callable[[np.ndarray, np.ndarray], float],
        gamma: Optional[float] = None,
        coef0: Optional[float] = None,
        degree: Optional[int] = None,
    ) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        '''
        Define the kernel function.
        '''

        if callable(ker):
            kernel = ker
        elif ker == 'linear':
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
        self._computed_kernel = self.kernel_func(self.X, self.X)

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

        # kernel_vals = self.kernel_func(self.X, X)
        # return sum(self.alphas[n] * self.y[n] * kernel_vals[n, :] for n in range(self.N))

        # Same as above 2 lines, but faster
        computed_kernel = self._computed_kernel if np.array_equal(X, self.X) else self.kernel_func(self.X, X)
        return (self.alphas * self.y) @ computed_kernel

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
            An array of shape (n_samples,) of classification results.
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

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        str_repr = f'{self.__class__.__name__}('
        for arg, arg_dflt in self._arg_defaults.items():
            if (arg_val := getattr(self, arg)) != arg_dflt:
                arg_str = f'\'{arg_val}\'' if isinstance(arg_val, str) else str(arg_val)
                str_repr += f'{arg}=' + arg_str + ', '
        if str_repr.endswith(', '):
            str_repr = str_repr[:-2]
        str_repr += ')'
        return str_repr


class QSVMGroup:
    def __init__(
        self,
        qsvm_params: dict[str, Any],
        S: int,
        M: int,
        multiplier: float,
        balance_classes: bool,
        num_workers: int = 1,
    ) -> None:
        '''
        A QSVMGroup represents a set of S QSVMs train random M-sample subsets of the training set.

        Parameters
        ----------
        qsvm_params : dict[str, Any]
            A dictionary of keyword parameters to initialize each QSVM.
        S : int
            The number of classifiers to consider.
        M : int
            The size of the subsets each QSVM is trained on.
        balance_classes : bool
            Whether or not to balance the number of points from each class in every M-sample subset.
        num_workers : int, optional
            The number of threads to launch for parallelization.
        '''

        self.qsvm_params = qsvm_params
        self.S = S
        self.M = M
        self.balance_classes = balance_classes
        self.num_workers = num_workers
        self.multiplier = multiplier

        self._x_subsets: list[np.ndarray]
        self._y_subsets: list[np.ndarray]
        self._trained_qsvms: list[QSVM]
        self._weights: np.ndarray

    def _define_data_subsets(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Define the M-sample subsets on which to train each QSVM.
        Splits the training sets into self.S subsets of size self.M
        '''

        data_indices = np.arange(X.shape[0])
        cls1_indices = data_indices[y == 1] if self.balance_classes else None
        cls2_indices = data_indices[y == -1] if self.balance_classes else None

        self._x_subsets = []
        self._y_subsets = []
        for _ in range(self.S):
            if self.balance_classes:
                np.random.shuffle(cls1_indices)
                np.random.shuffle(cls2_indices)
                cls1_size = self.M // 2
                cls2_size = self.M - cls1_size
                subset_indices = np.hstack((cls1_indices[:cls1_size], cls2_indices[:cls2_size]))
                np.random.shuffle(subset_indices)
            else:
                np.random.shuffle(data_indices)
                subset_indices = data_indices[: self.M]

            self._x_subsets.append(X[subset_indices])
            self._y_subsets.append(y[subset_indices])

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'QSVMGroup':
        '''
        Fit the QSVMGroup on input data X and corresponding labels y.

        Parameters
        ----------
        X : np.ndarray
            An array of shape (n_samples, n_features) of training data.
        y : np.ndarray
            An array of shape (n_samples,) of binary class labels for the training data.
        '''

        self._define_data_subsets(X, y)

        def train_and_score(subsets: tuple[np.ndarray, np.ndarray]) -> tuple[QSVM, float]:
            x_subset, y_subset = subsets
            qsvm = QSVM(**self.qsvm_params)
            qsvm.fit(x_subset, y_subset)
            acc = qsvm.score(X, y)
            return qsvm, acc

        # Use ThreadPoolExecutor to parallelize the process
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(train_and_score, zip(self._x_subsets, self._y_subsets)))

        # Unpack the results into _trained_qsvms and accuracies
        self._trained_qsvms, accs = zip(*results)

        # self._trained_qsvms = []
        # accs = []
        # for x_subset, y_subset in zip(self._x_subsets, self._y_subsets):
        #     qsvm = QSVM(**self.qsvm_params)
        #     qsvm.fit(x_subset, y_subset)
        #     accs.append(qsvm.score(X, y))
        #     self._trained_qsvms.append(qsvm)

        accs = np.asarray(accs)
        self._weights = softmax(self.multiplier * accs).reshape(-1, 1)

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
            An array of shape (n_samples,) of classification results.
        '''

        # Get predictions for each QSVM
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(lambda qsvm: qsvm.predict(X), self._trained_qsvms))
        results = np.asarray(results)

        # Combine predictions as a weighted sum
        preds = (self._weights * results).sum(axis=0)
        preds = np.sign(preds).astype(np.int8)

        # Set unclassified samples arbitrarily
        if np.any(preds == 0):
            warnings.warn(f'{sum(preds == 0)} samples lie on the decision boundary. Arbitrarily assigning class 1.')
            preds[preds == 0] = 1  # Go with class 1 if we don't know

        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        '''
        Computes the accuracy of the QSVMGroup on inputs X against targets y.

        Parameters
        ----------
        X : np.ndarray
            The input data on which to run the QSVM.
        y : np.ndarray
            The target classes.

        Returns
        -------
        float
            The accuracy of the QSVMGroup.
        '''

        y = y.astype(np.int8)

        preds = self.predict(X)  # Normalization happens here
        acc = (preds == y).sum() / len(y)
        return acc


if __name__ == '__main__':
    import time

    sz = 1000
    x = np.random.rand(sz, 4)
    y = 2 * np.random.randint(0, 2, sz) - 1
    qsvmgroup = QSVMGroup({}, S=20, M=50, balance_classes=True, num_workers=8)

    t = time.perf_counter()
    qsvmgroup.fit(x, y)
    print(time.perf_counter() - t)

    x2 = np.random.rand(1000 * sz, 4)

    t = time.perf_counter()
    qsvmgroup.predict(x2)
    print(time.perf_counter() - t)
