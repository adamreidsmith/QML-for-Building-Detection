import warnings
import time
import numbers
from typing import Optional, Any
from collections.abc import Callable
from functools import partial

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler, SteepestDescentSampler
from dwave.system import LeapHybridSampler, DWaveSampler, AutoEmbeddingComposite
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel

from sklearn.base import BaseEstimator, ClassifierMixin, _fit_context
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn.utils._param_validation import Interval, Options
from sklearn.utils.multiclass import check_classification_targets, type_of_target
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError


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


class QSVM(ClassifierMixin, BaseEstimator):
    '''
    Quantum Support Vector Machine

    References
    ----------
    [1] D. Willsch, M. Willsch, H. De Raedt, and K. Michielsen. "Support vector machines on the d-wave quantum
        annealer". Computer Physics Communications, 248:107006, 2020.
        https://www.sciencedirect.com/science/article/pii/S001046551930342X.
    '''

    _dwave_sampler: Optional[DWaveSampler] = None
    _hybrid_sampler: Optional[LeapHybridSampler] = None

    _parameter_constraints = {
        'kernel': [Options(str, {'linear', 'rbf', 'poly', 'sigmoid'}), callable],  # valid string options or callable
        'B': [Interval(numbers.Integral, 2, None, closed='left')],  # integer larger than 1
        'P': [Interval(numbers.Integral, None, None, closed='neither')],  # any integer
        'K': [Interval(numbers.Integral, 1, None, closed='left')],  # positive integer
        'zeta': [Interval(numbers.Real, 0.0, None, closed='left')],  # non-negative float
        'gamma': [Interval(numbers.Real, 0.0, None, closed='left')],  # non-negative float
        'coef0': [numbers.Real],  # any float
        'degree': [Interval(numbers.Integral, 0, None, closed='left')],  # non-negative integer
        'sampler': [Options(str, {'qa', 'simulate', 'steepest_descent', 'tabu', 'hybrid'})],  # valid string options
        'num_reads': [Interval(numbers.Integral, 1, None, closed='left')],  # positive integer
        'hybrid_time_limit': [Interval(numbers.Real, 0.0, None, closed='neither')],  # positive float
        'normalize': ['boolean'],  # boolean
        'multi_class_strategy': [Options(str, {'ovo', 'ovr'})],  # valid string options
    }

    def __init__(
        self,
        kernel: str | Callable[[np.ndarray, np.ndarray], float] = 'rbf',
        B: int = 2,
        P: int = 0,
        K: int = 4,
        zeta: float = 1.0,
        gamma: float = 1.0,
        coef0: float = 0.0,
        degree: int = 3,
        sampler: str = 'steepest_descent',
        num_reads: int = 100,
        hybrid_time_limit: float = 3,
        normalize: bool = True,
        multi_class_strategy: str = 'ovo',
    ) -> None:
        '''
        Parameters
        ----------
        kernel : str | Callable[[np.ndarray, np.ndarray], float]
            Specifies the kernel type to be used in the algorithm. If a string, one of 'linear', 'rbf', 'poly',
            or 'sigmiod'. If a Callable, it must accept two 2D numpy arrays X and Y as arguments, and return a
            2D array of shape (X.shape[0], Y.shape[0]). Default is 'rbf'.
        B : int
            Base to use in the binary encoding of the QSVM coefficients. See equation (10) in [1].
            Default is 2.
        P : int
            The shift parameter in the encoding exponent that allows for negative exponents.
            Default is 0.
        K : int
            The number of binary encoding variables to use in the encoding of the QSVM coefficients. See equation
            (10) in [1]. Default is 3.
        zeta : float
            A parameter of the QSVM which enforces one of the constraints of a support vector machine in the QSVM
            QUBO model. See equation (11) in [1]. Default is 1.0.
        gamma : float
            Kernel coefficient for 'rbf', 'poly', and 'sigmoid' kernels. Must be non-negative.
            Default is 1.0.
        coef0 : float
            Independent term in kernel function. It is only significant in 'poly' and 'sigmoid' kernels.
            Default is 0.0.
        degree : int
            Degree of the polynomial kernel function ('poly'). Must be non-negative. Ignored by all other kernels.
            Default is 3.
        sampler : str
            The sampler used for annealing. One of 'qa', 'simulate', 'steepest_descent', 'tabu', or 'hybrid'. Only 'qa'
            and 'hybrid' use real quantum hardware. 'qa' will fail if the problem is too large to easily embed on the
            quantum annealer.  Default is 'steepest_descent'.
        num_reads : int
            Number of reads of the quantum or simulated annealer to compute the QSVM solution.
            Default is 100.
        hybrid_time_limit : float
            The time limit in seconds for the hybrid solver. Default is 3.
        normalize : bool
            Whether or not to normalize input data. Default is True.
        multi_class_strategy : str
            The strategy to use if targets have more than two classes. One of 'ovo' for one-vs-one or 'ovr for
            one-vs-rest. Default is 'ovo'.
        '''

        self.B = B  # Base of the encoding
        self.P = P  # Encoding exponent shift
        self.K = K  # Number of encoding variables
        self.zeta = zeta  # Constraint term coefficient

        # Kernel parameters
        self.gamma = gamma
        self.coef0 = coef0
        self.degree = degree
        self.kernel = kernel

        # Annealing method parameters
        self.sampler = sampler
        self.num_reads = num_reads
        self.hybrid_time_limit = hybrid_time_limit

        # Other parameters
        self.normalize = normalize
        self.multi_class_strategy = multi_class_strategy

    def _define_kernel(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        '''
        Define the kernel function.
        '''

        if callable(self.kernel):
            kernel = self.kernel
        elif self.kernel == 'linear':
            kernel = linear_kernel
        elif self.kernel == 'rbf':
            kernel = partial(rbf_kernel, gamma=self.gamma)
        elif self.kernel == 'sigmoid':
            kernel = partial(sigmoid_kernel, gamma=self.gamma, coef0=self.coef0)
        elif self.kernel == 'poly':
            kernel = partial(polynomial_kernel, gamma=self.gamma, coef0=self.coef0, degree=self.degree)
        else:
            raise ValueError(f'Unknown kernel: {self.kernel}')
        return kernel

    @_fit_context(prefer_skip_nested_validation=True)
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

        # Check that inputs are of the right shape and that all parameters
        # are valid (as defined in cls._parameter_constraints).
        # This casts X and y to numpy arrays.
        X, y = self._validate_data(X, y, ensure_min_samples=2)
        X: np.ndarray
        y: np.ndarray

        # Make sure we have a classification task
        check_classification_targets(y)

        # Check that targets are binary or multiclass
        y_type = type_of_target(y)
        if y_type not in ['binary', 'multiclass']:
            raise ValueError('y must be a 1D array of classes. Classes can be represetned by integers or strings.')
        self.is_multi_class_ = y_type == 'multiclass'

        # Required to be set before multiclass output is handled
        self.classes_ = np.unique(y)  # Store the unique classes

        if self.is_multi_class_:
            if self.multi_class_strategy == 'ovr':
                self.multi_class_model_ = OneVsRestClassifier(self)
            elif self.multi_class_strategy == 'ovo':
                self.multi_class_model_ = OneVsOneClassifier(self)
            else:
                raise ValueError(f'Unknown multi_class_strategy: {self.multi_class_strategy}')
            self.multi_class_model_.fit(X, y)  # Fit the multi-class model
            return self

        # Binary classification
        self.B_ = float(self.B)  # B must be a float for array exponentiation
        self.classes_binary_ = np.unique(y)  # Store the unique classes
        X_ = self._normalize(X, train=True)  # Normalize the data
        y_ = self._binarize_y(y)  # Convert y from arbitrary class labels to ±1

        if hasattr(self, 'bias_') and np.array_equal(self.X_, X_) and np.array_equal(self.y_, y_):
            return self

        self.X_ = X_
        self.y_ = y_
        self.N_ = self.X_.shape[0]  # Number of train points

        # Compute the kernel matrix from the train data
        self.kernel_func_ = self._define_kernel()
        self.computed_kernel_ = self.kernel_func_(self.X_, self.X_)

        # self.qubo_ = self._compute_qubo()
        self.qubo_ = self._compute_qubo_fast()

        sample = self._run_sampler()
        self.alphas_ = self._compute_qsvm_coefficients(sample)
        self.bias_ = self._compute_bias()
        return self

    def _binarize_y(self, y: np.ndarray, reverse: bool = False) -> np.ndarray:
        '''
        Returns an array of the same shape as y, but with classes mapped to (-1, 1).
        If `reverse=True`, map an array of (-1, 1) back to the original classes.
        '''

        if np.all(np.isin(self.classes_binary_, (-1, 1))):
            return y.astype(int)
        if reverse:
            return np.where(y == -1, self.classes_binary_[0], self.classes_binary_[1])
        return np.where(y == self.classes_binary_[0], -1, 1)

    def _normalize(self, X: np.ndarray, train: bool = False) -> np.ndarray:
        '''
        Normalize the input X w.r.t. the train data. If train=True, the mean and standard deviation are computed from X.
        '''

        if not self.normalize:
            return X
        if train:
            self.normalization_mean_ = np.mean(X, axis=0)
            self.normalization_stdev_ = np.std(X, axis=0)
        return (X - self.normalization_mean_) / self.normalization_stdev_

    def _compute_qubo(self) -> np.ndarray:
        '''
        Compute the QUBO matrix associated to the QSVM problem from the precomputed kernel
        and the targets y.
        '''

        coeff_matrix = self.computed_kernel_ + self.zeta  # N x N array
        for n in range(self.N_):
            for m in range(self.N_):
                coeff_matrix[n, m] *= 0.5 * self.y_[n] * self.y_[m]

        base_power = np.empty((self.K, self.K))  # K x K array
        for k in range(self.K):
            for j in range(self.K):
                base_power[k, j] = self.B_ ** (k + j - 2 * self.P)

        # Formulate the QUBO as an upper triangular matrix
        qubo = {}
        for n in range(self.N_):
            for m in range(self.N_):
                for k in range(self.K):
                    for j in range(self.K):
                        kk = self.K * n + k
                        jj = self.K * m + j
                        if kk < jj:
                            qubo[(kk, jj)] = 2 * coeff_matrix[n, m] * base_power[k, j]
                        elif kk == jj:
                            qubo[(kk, jj)] = coeff_matrix[n, m] * base_power[k, j] - self.B_ ** (k - self.P)
        return qubo

    def _compute_qubo_fast(self) -> np.ndarray:
        '''
        Compute the QUBO matrix associated to the QSVM problem from the precomputed kernel
        and the targets y. This performs the same operation as self._compute_qubo but utilizes
        numpy array methods for parallelization.
        '''

        # Compute the coefficients without the base exponent or diagonal adjustments
        y_expanded = self.y_[:, np.newaxis]
        coeff_matrix = 0.5 * y_expanded @ y_expanded.T * (self.computed_kernel_ + self.zeta)

        # Base exponent values
        k = np.arange(self.K).reshape(self.K, 1) - self.P
        j = np.arange(self.K) - self.P
        base_power = self.B_ ** (k + j)

        # Compute the complete matrix using the Kronecker product
        qubo_array = np.kron(coeff_matrix, base_power)

        # Adjust diagonal elements
        diag_indices = np.arange(self.K * self.computed_kernel_.shape[0])
        qubo_array[diag_indices, diag_indices] -= np.tile(self.B_**j, self.computed_kernel_.shape[0])

        # Define the QUBO matrix, which is given by
        #     { Q_tilde[i, j] + Q_tilde[j, i]   if i < j
        # Q = { Q_tilde[i, j]                   if i == j
        #     { 0                               otherwise
        lower_triangle_indices = np.tril_indices_from(qubo_array, -1)
        upper_triangle_indices = np.triu_indices_from(qubo_array, 1)
        qubo_array[lower_triangle_indices] = 0
        qubo_array[upper_triangle_indices] *= 2

        qubo = {(i, j): qubo_array[(i, j)] for i in range(self.K * self.N_) for j in range(i, self.K * self.N_)}
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

        sample_set = sampler.sample_qubo(self.qubo_, num_reads=self.num_reads)
        return sample_set.first.sample

    def _run_hybrid_sampler(self) -> dict[int, int]:
        '''
        Run the Leap hybrid sampler on self.qubo.
        '''

        if self._hybrid_sampler is None:
            self._set_hybrid_sampler()
        sample_set = self._hybrid_sampler.sample_qubo(self.qubo_, time_limit=self.hybrid_time_limit)
        return sample_set.first.sample

    @classmethod
    def _set_hybrid_sampler(cls) -> None:
        '''
        Set the hybrid sampler as a class attribute so we can reuse the instance over multiple QSVM instances.
        This prevents the creation of too many threads when many QSVM instances are initialized.
        '''

        cls._hybrid_sampler = LeapHybridSampler()

    @classmethod
    def _set_dwave_sampler(cls) -> None:
        '''
        Set the DWave sampler as a class attribute so we can reuse the instance over multiple QSVM instances.
        This prevents the creation of too many threads when many QSVM instances are initialized.
        '''

        cls._dwave_sampler = DWaveSampler()

    def _compute_qsvm_coefficients(self, sample: dict[int, int]) -> np.ndarray:
        '''
        Compute the coefficients of the QSVM from the annealing solution. Decodes the binary variable encoding.
        '''

        alphas = np.empty(self.N_)
        powers_of_b = self.B_ ** (np.arange(self.K) - self.P)
        for n in range(self.N_):
            alphas[n] = sum(powers_of_b[k] * sample[self.K * n + k] for k in range(self.K))
        return alphas

    def _compute_bias(self, n_biases: int = 100) -> float:
        '''
        Compute the bias which best fits the training data.
        This checks for a bias in the range [-results.max(), -results.min()] using `n_biases` tests
        and returns the one which maximizes classification accuracy of the training data.
        '''

        results = self._decision_function_no_bias(self.X_)

        def compute_acc(bias: float) -> float:
            preds = results + bias
            preds = np.where(preds > 0, 1, -1)
            acc = (preds == self.y_).sum() / self.y_.shape[0]
            return acc

        biases = np.linspace(-results.max(), -results.min(), n_biases)  # Biases to test
        accuracies = np.fromiter(map(compute_acc, biases), dtype=np.float64)  # Compute the accuracies for each bias
        best_acc_indices = (accuracies == accuracies.max()).nonzero()[0]  # Find indices of biases with best accuracy
        best_acc_index = best_acc_indices[len(best_acc_indices) // 2]  # Choose the middle bias index
        return biases[best_acc_index]

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

        check_is_fitted(self)

        if self.is_multi_class_:
            return self.multi_class_model_.decision_function(X)

        # Normalize the input data w.r.t. the training data
        X = self._normalize(X)

        return self._decision_function_no_bias(X) + self.bias_

    def _decision_function_no_bias(self, X: np.ndarray) -> np.ndarray:
        '''
        The decision function without the bias term. This is implemented separately from self.decision_function
        for use with bias optimization.
        '''

        # kernel_vals = self.kernel_func(self.X, X)
        # return sum(self.alphas[n] * self.y[n] * kernel_vals[n, :] for n in range(self.N))

        # Same as above 2 lines, but faster
        computed_kernel = self.computed_kernel_ if np.array_equal(X, self.X_) else self.kernel_func_(self.X_, X)
        return (self.alphas_ * self.y_) @ computed_kernel

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

        check_is_fitted(self)

        if self.is_multi_class_:
            return self.multi_class_model_.predict(X)

        X: np.ndarray = self._validate_data(X, reset=False)

        # Compute the predictions
        results = self.decision_function(X)
        preds = np.where(results > 0, 1, -1)

        preds = self._binarize_y(preds, reverse=True)
        return preds

    def __sklearn_is_fitted__(self) -> bool:
        # is_multi_class_ is the first instance attribute set in the fit method,
        # so if it's not present, its definitely not fit
        if not hasattr(self, 'is_multi_class_'):
            return False
        # If we are performing multiclass classification, check if the underlying
        # multiclass model is fit
        if self.is_multi_class_:
            try:
                check_is_fitted(self.multi_class_model_)
            except NotFittedError:
                return False
            else:
                return True
        # bias_ is the last attribute defined in the fit method,
        # so if it's present, the classifier has been successfully fit
        return hasattr(self, 'bias_')

    def __sklearn_clone__(self) -> 'QSVM':
        # The `get_params` method is defined in BaseEstimator and returns a dict of
        # the current values all params set in __init__
        return type(self)(**self.get_params())

    def __call__(self, *args, **kwargs):
        # Calling the instance calls the predict method
        return self.predict(*args, **kwargs)

    def _more_tags(self):
        return {'non_deterministic': True, 'requires_y': True}


if __name__ == '__main__':
    from sklearn.utils.estimator_checks import check_estimator

    # Occasionally checks fail since QSVM solution is non-deterministic
    check_estimator(QSVM())
