import os
import sys
import time
import numbers
import warnings
import logging
from typing import Optional, Any
from collections.abc import Callable, Iterator
from functools import partial

import numpy as np
import dimod
from dimod import BinaryQuadraticModel
from dwave.samplers import SimulatedAnnealingSampler, TabuSampler, SteepestDescentSampler
from dwave.system import LeapHybridSampler, DWaveSampler, DWaveCliqueSampler, AutoEmbeddingComposite
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel, polynomial_kernel, sigmoid_kernel
from tqdm import tqdm

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

    _dwave_sampler: Optional[DWaveSampler | DWaveCliqueSampler | LeapHybridSampler] = None

    _parameter_constraints = {
        'kernel': [Options(str, {'linear', 'rbf', 'poly', 'sigmoid'}), callable],  # valid string options or callable
        'B': [Interval(numbers.Integral, 2, None, closed='left')],  # integer larger than 1
        'P': [Interval(numbers.Integral, None, None, closed='neither')],  # any integer
        'K': [Interval(numbers.Integral, 1, None, closed='left')],  # positive integer
        'zeta': [Interval(numbers.Real, 0.0, None, closed='left')],  # non-negative float
        'gamma': [Interval(numbers.Real, 0.0, None, closed='left')],  # non-negative float
        'coef0': [numbers.Real],  # any float
        'degree': [Interval(numbers.Integral, 0, None, closed='left')],  # non-negative integer
        'sampler': [
            Options(str, {'qa', 'qa_clique', 'simulate', 'steepest_descent', 'tabu', 'hybrid'})
        ],  # valid string options
        'num_reads': [Interval(numbers.Integral, 1, None, closed='left')],  # positive integer
        'hybrid_time_limit': [Interval(numbers.Integral, 1, None, closed='left'), None],  # positive integer or None
        'normalize': ['boolean'],  # boolean
        'multi_class_strategy': [Options(str, {'ovo', 'ovr'})],  # valid string options
        'threshold': [Interval(numbers.Real, 0.0, None, closed='left')],  # non-negative float
        'threshold_strategy': [Options(str, {'absolute', 'relative'})],  # valid string options
        'optimize_memory': ['boolean'],  # boolean
        'max_annealing_tries': [Interval(numbers.Integral, 1, None, closed='left')],  # positive integer
        'retry_delay': [Interval(numbers.Real, 0.0, None, closed='left')],  # non-negative float
        'fail_to_classical': ['boolean'],  # boolean
        'dwave_api_token': [str, None],  # string or None
        'verbose': ['boolean'],  # boolean
    }

    def __init__(
        self,
        kernel: str | Callable[[np.ndarray, np.ndarray], float | np.ndarray] = 'rbf',
        B: int = 2,
        P: int = 0,
        K: int = 4,
        zeta: float = 1.0,
        gamma: float = 1.0,
        coef0: float = 0.0,
        degree: int = 3,
        sampler: str = 'steepest_descent',
        num_reads: int = 100,
        hybrid_time_limit: Optional[int] = 3,
        normalize: bool = True,
        multi_class_strategy: str = 'ovo',
        threshold: float = 0.0,
        threshold_strategy: str = 'absolute',
        optimize_memory: bool = True,
        max_annealing_tries: int = 2,
        retry_delay: float = 2.0,
        fail_to_classical: bool = True,
        dwave_api_token: Optional[str] = None,
        verbose: bool = False,
    ) -> None:
        '''
        Parameters
        ----------
        kernel : str | Callable[[np.ndarray, np.ndarray], float]
            Specifies the kernel type to be used in the algorithm. If a string, one of 'linear', 'rbf', 'poly',
            or 'sigmiod'. If a Callable, it must accept either two 2D or 2 1D numpy arrays X and Y as arguments,
            If 2D, it must return the 2D kernel matrix of shape (X.shape[0], Y.shape[0]).  If 1D, it must return
            the kernel of X with Y as a float. Default is 'rbf'.
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
            The sampler used for annealing. One of 'qa', 'qa_clique', 'simulate', 'steepest_descent', 'tabu', or
            'hybrid'. Only 'qa', 'qa_clique', and 'hybrid' use real quantum hardware. 'qa' will fail if the problem
            is too large to easily embed on the quantum annealer.  Default is 'steepest_descent'.
        num_reads : int
            Number of reads of the quantum or simulated annealer to compute the QSVM solution.
            Default is 100.
        hybrid_time_limit : int | None
            The time limit in seconds for the hybrid solver. Default is 3.
        normalize : bool
            Whether or not to normalize input data. Default is True.
        multi_class_strategy : str
            The strategy to use if targets have more than two classes. One of 'ovo' for one-vs-one or 'ovr for
            one-vs-rest. Default is 'ovo'.
        threshold : float
            A threshold to use in the QSVM QUBO model.  Biases and coupling strengths with absolute value below this
            threshold are set to zero to reduce the size of the QUBO and speed up computation. Default is 0.0.
        threshold_strategy : str
            The strategy to use when setting the threshold. One of 'absolute' or 'relative'. 'absolute' sets all QUBO
            elements with absolute value less than `threshold` to zero. 'relative' sets the smallest
            `num_nonzero_qubo_elements * threshold` QUBO elements to zero. 'relative' is not available when
            `optimize_memory=True`. Default is 'absolute'.
        optimize_memory : bool
            Whether or not to optimize the memory usage at the expense of computation time. Default is True.
        max_annealing_tries: int
            The maximum number of times to try running the annealing method. Default is 2.
        retry_delay : float
            The delay in seconds between retries of the annealing method. Only applies if max_annealing_tries > 1.
            Default is 2.0.
        fail_to_classical : bool
            If True and sampling a QUBO on a quantum device fails, sampling is re-run using a classical solver.
            Default is True.
        dwave_api_token : str | None
            An API token for a D-Wave Leap account. This is necessary to run annealing on quantum hardware (sampler is
            'qa', 'qa_clique', or 'hybrid'). In this case, the environment variable DWAVE_API_TOKEN is set to the
            provided token. It is not necessary if quantum computation is being simulated (sampler is 'simulate',
            'tabu', or 'steepest_descent'). Default is None.
        verbose : bool
            Whether or not to print progress messages. Default is False.
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
        self.dwave_api_token = dwave_api_token
        self.threshold = threshold
        self.threshold_strategy = threshold_strategy
        self.optimize_memory = optimize_memory
        self.max_annealing_tries = max_annealing_tries
        self.retry_delay = retry_delay
        self.fail_to_classical = fail_to_classical
        self.verbose = verbose

    def _configure_logger(self) -> None:
        self._logger = logging.getLogger(self.__class__.__name__)
        handler = logging.StreamHandler(sys.stdout)  # Log to standard output
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        if not self._logger.hasHandlers():
            self._logger.addHandler(handler)
        if self.verbose:
            self._logger.setLevel(logging.INFO)
        else:
            self._logger.setLevel(logging.WARNING)

    def _define_kernel(self) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
        '''
        Define the kernel function.
        '''

        if callable(self.kernel):
            kernels = self.kernel, self.kernel
        elif self.kernel == 'linear':
            kernels = linear_kernel, self._linear_kernel1D
        elif self.kernel == 'rbf':
            kernels = partial(rbf_kernel, gamma=self.gamma), partial(self._rbf_kernel1D, gamma=self.gamma)
        elif self.kernel == 'sigmoid':
            kernels = (
                partial(sigmoid_kernel, gamma=self.gamma, coef0=self.coef0),
                partial(self._sigmoid_kernel1D, gamma=self.gamma, coef0=self.coef0),
            )
        elif self.kernel == 'poly':
            kernels = (
                partial(polynomial_kernel, gamma=self.gamma, coef0=self.coef0, degree=self.degree),
                partial(self._polynomial_kernel1D, gamma=self.gamma, coef0=self.coef0, degree=self.degree),
            )
        else:
            raise ValueError(f'Unknown kernel: {self.kernel}')
        return kernels

    @staticmethod
    def _linear_kernel1D(X: np.ndarray, Y: np.ndarray) -> float:
        '''
        Compute the linear kernel of a single pair of samples.
        '''

        return X @ Y

    @staticmethod
    def _rbf_kernel1D(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
        '''
        Compute the RBF kernel of a single pair of samples.
        '''

        return np.exp(-gamma * ((X - Y) ** 2).sum())

    @staticmethod
    def _sigmoid_kernel1D(X: np.ndarray, Y: np.ndarray, gamma: float, coef0: float) -> float:
        '''
        Compute the sigmoid kernel of a single pair of samples.
        '''

        return np.tanh(gamma * X @ Y + coef0)

    @staticmethod
    def _polynomial_kernel1D(X: np.ndarray, Y: np.ndarray, gamma: float, coef0: float, degree: int) -> float:
        '''
        Compute the polynomial kernel of a single pair of samples.
        '''

        return (gamma * X @ Y + coef0) ** degree

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

        self._configure_logger()

        self._logger.info('Fitting QSVM')

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
            raise ValueError('y must be a 1D array of classes. Classes can be represented by integers or strings.')
        self.is_multi_class_ = y_type == 'multiclass'

        # Required to be set before multiclass output is handled
        self.classes_ = np.unique(y)  # Store the unique classes

        if self.is_multi_class_:
            self._logger.info(f'Handling multi-class classification with strategy {self.multi_class_strategy}')
            if self.multi_class_strategy == 'ovr':
                self.multi_class_model_ = OneVsRestClassifier(self)
            elif self.multi_class_strategy == 'ovo':
                self.multi_class_model_ = OneVsOneClassifier(self)
            else:
                raise ValueError(f'Unknown multi_class_strategy: {self.multi_class_strategy}')
            self.multi_class_model_.fit(X, y)  # Fit the multi-class model
            return self

        if self.sampler in ('hybrid', 'qa', 'qa_clique') and self.dwave_api_token is not None:
            self._logger.info(f'Setting D-Wave API token')
            os.environ['DWAVE_API_TOKEN'] = self.dwave_api_token

        # Binary classification
        self.B_ = float(self.B)  # B must be a float for array exponentiation
        self.classes_binary_ = np.unique(y)  # Store the unique classes
        X_ = self._normalize(X, train=True)  # Normalize the data
        y_ = self._binarize_y(y)  # Convert y from arbitrary class labels to ±1

        if hasattr(self, 'bias_') and np.array_equal(self.X_, X_) and np.array_equal(self.y_, y_):
            self._logger.info('Already fit QSVM with these parameters. Returning existing model.')
            return self

        self.X_ = X_
        self.y_ = y_
        self.N_ = self.X_.shape[0]  # Number of train points

        self.kernel_func_2D_, self.kernel_func_1D_ = self._define_kernel()
        if self.optimize_memory:
            if self.threshold_strategy == 'relative':
                raise ValueError('Relative threshold strategy not available when optimize_memory=True')
        else:
            self.computed_kernel_ = self.kernel_func_2D_(self.X_, self.X_)
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
        self._logger.info(f'{"De-binarizing" if reverse else "Binarizing"} targets')
        if reverse:
            return np.where(y == -1, self.classes_binary_[0], self.classes_binary_[1])
        return np.where(y == self.classes_binary_[0], -1, 1)

    def _normalize(self, X: np.ndarray, train: bool = False) -> np.ndarray:
        '''
        Normalize the input X w.r.t. the train data. If train=True, the mean and standard deviation are computed from X.
        '''

        if not self.normalize:
            return X
        self._logger.info('Normalizing data')
        if train:
            self.normalization_mean_ = np.mean(X, axis=0)
            self.normalization_stdev_ = np.std(X, axis=0)
        return (X - self.normalization_mean_) / self.normalization_stdev_

    def _qubo_generator(self) -> Iterator[tuple[tuple[int, int], float]]:
        '''
        Generate QUBO elements on-the-fly to save memory.
        '''

        k = np.arange(self.K).reshape(self.K, 1)
        j = np.arange(self.K)
        base_power = self.B_ ** (k + j - 2.0 * self.P)
        base_power_single = self.B_ ** (j - self.P)

        for n in range(self.N_):
            for m in range(n, self.N_):
                coeff = self.y_[n] * self.y_[m] * (self.kernel_func_1D_(self.X_[n], self.X_[m]) + self.zeta)
                for k in range(self.K):
                    for j in range(self.K):
                        ii = self.K * n + k
                        jj = self.K * m + j
                        if ii < jj:
                            yield (ii, jj), coeff * base_power[k, j]
                        elif ii == jj:
                            yield (ii, jj), 0.5 * coeff * base_power[k, j] - base_power_single[k]

    def _compute_qubo_fast(self) -> np.ndarray:
        '''
        Compute the QUBO matrix associated to the QSVM problem from the precomputed kernel
        and the targets y.
        '''

        self._logger.info('Computing QUBO explicitly')

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

        return qubo_array

    def _run_sampler(self) -> dict[int, int]:
        '''
        Run the hybrid sampler or the simulated annealing sampler. Returns a dict that maps the
        indices of the binary variables to their values.
        '''

        retry_args = dict(max_retries=self.max_annealing_tries, delay=self.retry_delay, warn=True)
        try:
            if self.sampler == 'hybrid':
                return retry(self._run_hybrid_sampler, **retry_args)
            return retry(self._run_pure_sampler, **retry_args)
        except Exception as e:
            if self.fail_to_classical and self.sampler in ('qa', 'qa_clique', 'hybrid'):
                self.sampler = 'steepest_descent'
                return retry(self._run_pure_sampler, **retry_args)
            else:
                raise e

    def _run_pure_sampler(self) -> dict[int, int]:
        '''
        Run a purely quantum or classical sampling method on the QUBO.
        '''

        if self.sampler == 'simulate':
            sampler = SimulatedAnnealingSampler()

        elif self.sampler == 'tabu':
            sampler = TabuSampler()

        elif self.sampler == 'steepest_descent':
            sampler = SteepestDescentSampler()

        elif self.sampler == 'qa':
            if not isinstance(self._dwave_sampler, AutoEmbeddingComposite):
                self._set_dwave_sampler()
            sampler = AutoEmbeddingComposite(self._dwave_sampler)

        elif self.sampler == 'qa_clique':
            if not isinstance(self._dwave_sampler, DWaveCliqueSampler):
                self._set_dwave_clique_sampler()
            sampler = self._dwave_sampler

        else:
            raise ValueError(f'Unknown pure sampler: {self.sampler}')

        return self._run_from_sampler(sampler, {'num_reads': self.num_reads})

    def _run_hybrid_sampler(self) -> dict[int, int]:
        '''
        Run the Leap hybrid sampler on the QUBO.
        '''

        if not isinstance(self._dwave_sampler, LeapHybridSampler):
            self._set_hybrid_sampler()
        return self._run_from_sampler(self._dwave_sampler, {'time_limit': self.hybrid_time_limit})

    def _run_from_sampler(
        self,
        sampler: dimod.Sampler,
        sampler_kwargs: dict[str, Any],
    ) -> dict[int, int]:
        '''
        Threshold the qubo and run a given sampler on the QUBO.
        '''

        if self.optimize_memory:
            bqm = self._generate_thresholded_bqm()
            self._logger.info('Sampling BQM')
            sample_set = sampler.sample(bqm, **sampler_kwargs)
        else:
            qubo = self._compute_thresholded_qubo()
            self._logger.info('Sampling QUBO')
            sample_set = sampler.sample_qubo(qubo, **sampler_kwargs)

        return sample_set.first.sample

    def _generate_thresholded_bqm(self) -> BinaryQuadraticModel:
        '''
        Build a BQM while thresholding QUBO elements on-the-fly to save memory.
        '''

        self._logger.info('Generating BQM')

        self.n_rejected_ = self.n_accepted_ = 0
        bqm = BinaryQuadraticModel(vartype=dimod.BINARY, dtype=np.float32)
        num_qubo_elements = self.N_ * self.K * (self.N_ * self.K + 1) / 2
        for (u, v), bias in tqdm(
            self._qubo_generator(), total=num_qubo_elements, disable=not self.verbose, desc='Generating QUBO'
        ):
            if u == v:
                bqm.add_linear(u, bias)
                continue
            if (b := abs(bias)) < self.threshold:
                self.n_rejected_ += b > 0
                continue
            self.n_accepted_ += 1
            bqm.add_quadratic(u, v, bias)

        return bqm

    def _compute_thresholded_qubo(self) -> np.ndarray:
        '''
        Explicitly compute the thresholded QUBO matrix.
        '''

        if self.threshold > 0:
            if self.threshold_strategy == 'absolute':
                upper_tri_indices = np.triu_indices_from(self.qubo_, 1)
                upper_tri_elements = np.abs(self.qubo_[upper_tri_indices])

                self.n_rejected_ = ((upper_tri_elements > 0) & (upper_tri_elements < self.threshold)).sum()
                self.n_accepted_ = (upper_tri_elements >= self.threshold).sum()

                qubo_thres = self.qubo_.copy()
                qubo_thres[upper_tri_indices] = np.where(
                    upper_tri_elements < self.threshold, 0, self.qubo_[upper_tri_indices]
                )
            else:
                qubo_thres = np.triu(self.qubo_, k=1)
                n_nonzero = np.count_nonzero(qubo_thres)

                self.n_rejected_ = int(n_nonzero * self.threshold)
                self.n_accepted_ = n_nonzero - self.n_rejected_

                triu_indices = np.triu_indices_from(self.qubo_, 1)
                unique_vals, counts = np.unique(np.abs(qubo_thres[triu_indices]), return_counts=True)
                cumsum_counts = np.cumsum(counts)
                threshold_index = np.searchsorted(cumsum_counts, self.n_rejected_, side='left')

                # Zero out all elements whose absolute value is less than or equal to unique_vals[threshold_index - 1]
                to_zero_mask1 = np.array([])
                if threshold_index > 0:
                    abs_triu_qubo_thres = np.abs(qubo_thres[triu_indices])
                    to_zero_mask1 = (abs_triu_qubo_thres <= unique_vals[threshold_index - 1]) & (
                        abs_triu_qubo_thres > 0
                    )
                    qubo_thres[triu_indices] = np.where(to_zero_mask1, 0, qubo_thres[triu_indices])

                to_keep = cumsum_counts[threshold_index] - self.n_rejected_

                # Zero out a random selection of the remaining n_rejected - already_zeroed elements equal to unique_vals[threshold_index]
                to_zero_mask2_flat = (np.abs(qubo_thres) == unique_vals[threshold_index]).flatten()
                to_zero_mask2_flat_set_false_indices = np.random.choice(
                    np.nonzero(to_zero_mask2_flat)[0], size=to_keep, replace=False
                )
                to_zero_mask2_flat[to_zero_mask2_flat_set_false_indices] = False
                to_zero_mask2 = to_zero_mask2_flat.reshape(qubo_thres.shape)
                qubo_thres[to_zero_mask2] = 0

                # assert (
                #     self.n_rejected_ == to_zero_mask1.sum() + to_zero_mask2.sum()
                # ), 'Number of rejected elements does not match'

                np.fill_diagonal(qubo_thres, np.diag(self.qubo_))  # Operates in-place

        else:
            qubo_thres = self.qubo_
            self.n_rejected_ = 0
            self.n_accepted_ = np.count_nonzero(self.qubo_[np.triu_indices_from(self.qubo_, 1)])

        return qubo_thres

    @classmethod
    def _set_hybrid_sampler(cls) -> None:
        '''
        Set the hybrid sampler as a class attribute so we can reuse the instance over multiple QSVM instances.
        This prevents the creation of too many threads when many QSVM instances are initialized.
        '''

        cls._dwave_sampler = LeapHybridSampler()

    @classmethod
    def _set_dwave_sampler(cls) -> None:
        '''
        Set the DWave sampler as a class attribute so we can reuse the instance over multiple QSVM instances.
        This prevents the creation of too many threads when many QSVM instances are initialized.
        '''

        cls._dwave_sampler = DWaveSampler()

    @classmethod
    def _set_dwave_clique_sampler(cls) -> None:
        '''
        Set the DWave sampler as a class attribute so we can reuse the instance over multiple QSVM instances.
        This prevents the creation of too many threads when many QSVM instances are initialized.
        '''

        cls._dwave_sampler = DWaveCliqueSampler()

    def _compute_qsvm_coefficients(self, sample: dict[int, int]) -> np.ndarray:
        '''
        Compute the coefficients of the QSVM from the annealing solution. Decodes the binary variable encoding.
        '''

        self._logger.info('Computing QSVM coefficients')

        alphas = np.empty(self.N_)
        powers_of_b = self.B_ ** (np.arange(self.K) - self.P)
        for n in range(self.N_):
            alphas[n] = sum(powers_of_b[k] * sample[self.K * n + k] for k in range(self.K))
        return alphas

    def _compute_bias(self, n_biases: int = 200) -> float:
        '''
        Compute the bias which best fits the training data.
        This checks for a bias in the range [-results.max(), -results.min()] using `n_biases` tests
        and returns the one which maximizes classification accuracy of the training data.
        '''

        self._logger.info('Computing optimal bias')

        results = self._decision_function_no_bias(self.X_)

        # Compute range for biases
        min_bias, max_bias = -results.max(), -results.min()
        if np.isclose(min_bias, max_bias):
            if np.all(results > 0):
                return max_bias - 1e-6  # Just below the minimum positive prediction
            else:
                return min_bias + 1e-6  # Just above the maximum negative prediction

        def compute_acc(bias: float) -> float:
            preds = results + bias
            preds = np.where(preds > 0, 1, -1)
            acc = (preds == self.y_).sum() / self.y_.shape[0]
            return acc

        biases = np.linspace(min_bias, max_bias, n_biases)  # Biases to test
        accuracies = np.fromiter(map(compute_acc, biases), dtype=np.float64)  # Compute the accuracies for each bias
        best_acc_indices = (accuracies == accuracies.max()).nonzero()[0]  # Find indices of biases with best accuracy
        best_acc_index = best_acc_indices[len(best_acc_indices) // 2]  # Choose the middle bias index
        bias = biases[best_acc_index]
        return bias

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
            self._logger.info('Computing multi-class decision function')
            return self.multi_class_model_.decision_function(X)

        # Normalize the input data w.r.t. the training data
        X = self._normalize(X)

        self._logger.info('Computing decision function')

        return self._decision_function_no_bias(X) + self.bias_

    def _decision_function_no_bias(self, X: np.ndarray, chunk_size: int = 2000) -> np.ndarray:
        '''
        The decision function without the bias term. This is implemented separately from self.decision_function
        for use with bias optimization.
        '''

        if self.optimize_memory:
            results = np.zeros(X.shape[0])
            for i in range(0, X.shape[0], chunk_size):  # Process in chunks of chunk_size
                chunk = X[i : i + chunk_size]
                kernel_chunk = self.kernel_func_2D_(self.X_, chunk)
                results[i : i + chunk_size] = (self.alphas_ * self.y_) @ kernel_chunk
        else:
            computed_kernel = self.computed_kernel_ if np.array_equal(X, self.X_) else self.kernel_func_2D_(self.X_, X)
            results = (self.alphas_ * self.y_) @ computed_kernel
        return results

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
            self._logger.info('Computing multi-class predictions')
            return self.multi_class_model_.predict(X)

        X: np.ndarray = self._validate_data(X, reset=False)

        self._logger.info('Computing predictions')

        # Compute the predictions
        results = self.decision_function(X)
        preds = np.where(results > 0, 1, -1)

        preds = self._binarize_y(preds, reverse=True)
        return preds

    def pre_fit_info(self, X: np.ndarray, y: np.ndarray) -> None:
        self._logger.info('Pre-fit information')

        X, y = self._validate_data(X, y, ensure_min_samples=2)
        check_classification_targets(y)

        if type_of_target(y) != 'binary':
            raise ValueError('Pre-fit only available for binary targets')

        try:
            largest_clique = DWaveCliqueSampler().largest_clique_size
        except ValueError:
            largest_clique = None

        N = X.shape[0]

        n_qubo_variables = N * self.K
        print(f'Number of QUBO variables: K * N = {self.K} * {N} = {n_qubo_variables}')
        if largest_clique is not None:
            print(f'Largest clique size: {largest_clique}')

            if n_qubo_variables <= largest_clique:
                print('QUBO samplable via clique sampler (sampler=\'qa_clique\')')
            else:
                print('No clique embedding possible. Use sampler=\'qa\' or sampler=\'hybrid\'')

        coef_max = self.B ** (2 * (self.K - 1) - self.P) * (1 + self.zeta)
        assume_str = '' if self.kernel in ('rbf', 'sigmoid') else ' (Assuming kernel(x, y)≤1)'
        print(f'Max coupling strength{assume_str}: {coef_max}')

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
