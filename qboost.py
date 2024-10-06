import os
import heapq
import warnings
from typing import Optional
from collections.abc import Sequence, Callable
from collections import defaultdict

import numpy as np
from dwave.samplers import SimulatedAnnealingSampler, SteepestDescentSampler, TabuSampler
from dwave.system import LeapHybridSampler, DWaveSampler, AutoEmbeddingComposite

from qsvm import retry


class QBoost:
    '''
    A class to implement the QBoost algorithm

    References
    ----------
    [1] Hartmut Neven, Vasil S. Denchev, Geordie Rose, and William G. Macready. Qboost: Large scale
        classifier training withadiabatic quantum optimization. In Steven C. H. Hoi and Wray Buntine,
        editors, Proceedings of the Asian Conference on Machine Learning, volume 25 of Proceedings of
        Machine Learning Research, pages 333-348, Singapore Management University, Singapore,
        04-06 Nov 2012. PMLR. https://proceedings.mlr.press/v25/neven12.html.
    '''

    _dwave_sampler: Optional[DWaveSampler] = None
    _hybrid_sampler: Optional[LeapHybridSampler] = None

    def __init__(
        self,
        weak_classifiers: Sequence[Callable[[np.ndarray], np.ndarray]],
        K: int,
        B: int,
        P: int,
        lbda: tuple[float, float, float],
        num_reads: int = 100,
        sampler: str = 'steepest_descent',
        dwave_api_token: Optional[str] = None,
    ) -> None:
        '''
        Parameters
        ----------
        weak_classifiers : Sequence[Callable[[np.ndarray], np.ndarray]]
            A sequence of classifiers. Each must map a 2d numpy array of samples with shape (n_samples, n_features)
            to a 1d numpy array of predictions with shape (n_samples,).
        K : int
            The bit depth for the coefficient encoding.
        B : int
            The base for the coefficient encoding.
        P : int
            The exponent shift for the coefficient encoding.
        lbda : tuple[float, float, float]
            A tuple of the form (start, stop, step) defining the regularization parameters.
        num_reads : int
            Number of reads for the quantum or classical annealer.
        sampler : str
            The sampler used for annealing. One of 'qa', 'simulate', 'steepest_descent', 'tabu', or 'hybrid'. Only 'qa'
            and 'hybrid' use real quantum hardware. 'qa' will fail if the problem is too large to easily embed on the
            quantum annealer.  Default is 'steepest_descent'.
        dwave_api_token : str | None
            An API token for a D-Wave Leap account. This is necessary to run annealing on quantum hardware (sampler is
            'hybrid' or 'qa'). In this case, the environment variable DWAVE_API_TOKEN is set to the provided token. It
            is not necessary if quantum computation is being simulated (sampler is 'simulate', 'tabu', or
            'steepest_descent'). Default is None.
        '''

        self.classifiers = list(weak_classifiers)

        # Encoding vars
        self.K = K
        self.B = float(B)
        self.P = P

        # Regularization vars
        self.lbda = lbda  # (lambda_start, lambda_stop, lambda_step)

        self.num_reads = num_reads
        self.sampler = sampler
        self.dwave_api_token = dwave_api_token

        self.eps = 1e-8  # Coefficients smaller than this are considered zero

        # Instantiated when `fit` is called
        self._strong_classifier_coeffs: Optional[list[float]] = None
        self._classifier_pool_indices: list[int]

    def kappa(self, lbda: float) -> float:
        # We must choose kappa > lbda / epsilon in order for L0 regularization to work properly.
        # Here, epsilon is the smallest positive value the QBoost coefficients can take
        return 2 * lbda * self.B**self.P

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> 'QBoost':
        '''
        Fit the QBoost model to the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training features of shape (n_samples, n_features)
        y_train : np.ndarray
            Training class labels of shape (n_samples,)
        x_val : np.ndarray | None, optional
            Validation features of shape (n_samples, n_features). If None, training data is used as validation data.
            Default is None.
        y_val : np.ndarray | None, optional
            Validation class labels of shape (n_samples,). If None, training data is used as validation data.
            Default is None.

        Returns
        -------
        self
        '''

        if self.sampler in ['hybrid', 'qa'] and self.dwave_api_token is not None:
            os.environ['DWAVE_API_TOKEN'] = self.dwave_api_token

        # Validate inputs and obtain classification results
        x_train, y_train = self._validate_inputs(x_train, y_train)
        n_train_samples = len(x_train)
        train_clf_results = np.empty((len(self.classifiers), n_train_samples))
        for i, clf in enumerate(self.classifiers):
            train_clf_results[i] = clf(x_train)

        # If we have validation data, classify it
        # Otherwise, use the training data as validation data
        if x_val is not None and y_val is not None:
            x_val, y_val = self._validate_inputs(x_val, y_val)
            n_val_samples = len(x_val)
            val_clf_results = np.empty((len(self.classifiers), n_val_samples))
            for i, clf in enumerate(self.classifiers):
                val_clf_results[i] = clf(x_val)
        else:
            x_val, y_val = x_train, y_train
            n_val_samples = len(x_val)
            val_clf_results = train_clf_results

        # d_inner is a distribution over training samples
        d_inner = np.full(n_train_samples, 1 / n_train_samples)
        T_inner = 0
        Q = len(self.classifiers)
        classifier_pool_indices = []

        best_coeffs, best_clf_indices, best_lbda, best_valid_error = None, None, None, float('inf')
        terminate = False
        while True:
            # From {h_i} (the pool of all weak classifiers) select the Q − T_inner weak classifiers that have the
            # smallest training error rates weighted by d_inner and add them to the pool {h_q}
            classification_results = [np.dot(d_inner, (clf_result - y_train) ** 2) for clf_result in train_clf_results]
            new_classifier_pool_indices = set(
                heapq.nsmallest(Q - T_inner, range(len(self.classifiers)), key=classification_results.__getitem__)
            )
            classifier_pool_indices = sorted(set(classifier_pool_indices) | new_classifier_pool_indices)

            # Stores the best known coefficients, classifiers, T_inner, lbda, and corresponding error
            best = (None, None, None, None, float('inf'))
            for lbda in np.arange(*self.lbda):
                # Define the qubo for the current regularization parameter and obtain the QBoost coefficients
                qubo = self._define_qubo(y_train, train_clf_results[classifier_pool_indices], lbda)
                vars = self._run_sampler(qubo)
                coeffs = self._decode_vars(vars, len(classifier_pool_indices))

                trial_T_inner = (np.abs(coeffs) > self.eps).sum()

                # Sample the strong classifier using the validation set and determine the validation error
                predictions = np.empty(n_val_samples)
                for s in range(n_val_samples):
                    predictions[s] = np.sign(np.dot(coeffs, val_clf_results[classifier_pool_indices][:, s]))
                trial_valid_error = (predictions != y_val).sum() / len(y_val)

                if trial_valid_error < best[-1]:
                    best = (coeffs, classifier_pool_indices, trial_T_inner, lbda, trial_valid_error)

            # If the validation error does not decrease, break from the loop
            old_valid_error, new_valid_error = best_valid_error, best[-1]
            if new_valid_error >= old_valid_error:
                terminate = True
            else:
                best_coeffs, best_clf_indices, T_inner, best_lbda, best_valid_error = best

            # Update the distribution over training samples d_inner
            for s in range(n_train_samples):
                d_inner[s] *= (
                    y_train[s] * np.dot(best_coeffs, train_clf_results[best_clf_indices][:, s]) / len(best_clf_indices)
                    - 1
                ) ** 2
            d_inner /= d_inner.sum()

            # Delete from the pool {h_q} the Q − T_inner weak learners for which the corresponding coefficient is 0
            indices_of_nonzero_coeffs = [i for i, coeff in enumerate(best_coeffs) if abs(coeff) > self.eps]
            best_coeffs = [best_coeffs[i] for i in indices_of_nonzero_coeffs]
            best_clf_indices = [best_clf_indices[i] for i in indices_of_nonzero_coeffs]
            classifier_pool_indices = [*best_clf_indices]
            assert len(best_clf_indices) == len(best_coeffs)

            if terminate:
                break

        if best_coeffs is None:
            raise RuntimeError('QBoost failed to find a strong classifier')

        self._strong_classifier_coeffs = best_coeffs
        self._classifier_pool_indices = classifier_pool_indices
        self._best_lbda = best_lbda
        return self

    def _validate_inputs(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        '''
        Validate the inputs to the fit method.
        '''

        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=np.int8)
        assert x.ndim == 2, 'x must be a 2-dimensional array with shape (n_samples, n_features)'
        assert y.ndim == 1, 'y must be a 1-dimensional array with shape (n_samples,)'
        assert len(x) == len(y), 'x and y must have the same length'
        assert np.all(np.isin(y, (1, -1))), 'y must be a binary array containing only ±1'
        return x, y

    def _define_qubo(
        self, y: np.ndarray, classification_results: np.ndarray, lbda: float
    ) -> dict[tuple[int, int], float]:
        '''
        Define the QUBO model.

        Parameters
        ----------
        y : np.ndarray
            Array of true classes of shape (n_samples,).
        classification_results : np.ndarray
            Array of results of running the classifiers on input samples. Should have shape (n_classifiers, n_samples).
        lbda : float
            Regularization parameter.
        kappa : float
            Regularization parameter.

        Returns
        -------
        dict[tuple[int, int], float]
            A mapping of tuples of variables to their biases and coupling strengths.
        '''

        Q = classification_results.shape[0]  # Number of classifiers
        classification_results /= Q  # Normalize the classification results

        S = classification_results.shape[1]  # Number of samples
        base_powers = self.B ** (np.arange(self.K) - self.P)
        kappa = self.kappa(lbda)
        qubo = defaultdict(float)

        # Linear (bias) term
        for q in range(Q):
            clf_term = np.dot(y, classification_results[q]) / S
            for k in range(self.K):
                ii = self.K * q + k
                qubo[(ii, ii)] -= 2 * base_powers[k] * clf_term

        # Quadratic (coupling) term
        for q in range(Q):
            for p in range(Q):
                clf_term = np.dot(classification_results[q], classification_results[p]) / S
                for k in range(self.K):
                    for j in range(self.K):
                        ii, jj = self.K * q + k, self.K * p + j
                        if ii > jj:
                            # Flip the indices to ensure we are adding to the upper triangular part
                            ii, jj = jj, ii
                        qubo[(ii, jj)] += base_powers[k] * base_powers[j] * clf_term

        # Regularization terms
        # Auxillary variables are indexed by indices from self.K * Q (inclusive) to (self.K + 1) * Q (exclusive).
        for q in range(Q):
            jj = self.K * Q + q
            qubo[(jj, jj)] += lbda
            for k in range(self.K):
                ii = self.K * q + k
                qubo_term = kappa * base_powers[k]
                qubo[(ii, ii)] += qubo_term
                qubo[(ii, jj)] -= qubo_term

        return dict(qubo)

    def _run_sampler(self, qubo: dict[tuple[int, int], float]) -> dict[int, int]:
        '''
        Run the hybrid sampler or the simulated annealing sampler. Returns a dict that maps the
        indices of the binary variables to their values.
        '''

        if self.sampler == 'hybrid':
            return retry(self._run_hybrid_sampler, max_retries=2, delay=1, warn=True, qubo=qubo)
        return retry(self._run_pure_sampler, max_retries=2, delay=1, warn=True, qubo=qubo)

    def _run_pure_sampler(self, qubo: dict[tuple[int, int], float]) -> dict[int, int]:
        '''
        Run a purely quantum or classical sampling method on the qubo.
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

        sample_set = sampler.sample_qubo(qubo, num_reads=self.num_reads)
        return sample_set.first.sample

    def _run_hybrid_sampler(self, qubo: dict[tuple[int, int], float]) -> dict[int, int]:
        '''
        Run the Leap hybrid sampler on the qubo.
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

    def _decode_vars(self, vars: dict[int, int], n_clf: int) -> list[float]:
        '''
        Compute the QBoost coefficients from their binary encodings.
        '''

        coeffs = []
        base_powers = self.B ** (np.arange(self.K) - self.P)
        coeffs = [sum(base_powers[k] * vars[self.K * q + k] for k in range(self.K)) for q in range(n_clf)]
        return coeffs

    def predict(self, x: np.ndarray) -> np.ndarray:
        '''
        Perform classification on samples in X.

        Parameters
        ----------
        X : np.ndarray
            An array of shape (n_samples, n_features) to classify.

        Returns
        -------
        np.ndarray
            An array if shape (n_samples,) containing the classification results.
        '''

        if self._strong_classifier_coeffs is None:
            raise RuntimeError(
                f'This QBoost instance is not fitted yet. '
                'Call \'fit\' with appropriate arguments before calling \'predict\''
            )

        # Only weak classifiers present in `self._classifier_pool_indices` are included in the strong classifier
        classifiers = (self.classifiers[j] for j in self._classifier_pool_indices)

        # Obtain classification results from the weak classifiers
        n_samples = x.shape[0]
        classification_results = np.empty((len(self._classifier_pool_indices), n_samples))
        for i, clf in enumerate(classifiers):
            classification_results[i] = clf(x)

        # Sample the strong classifier
        strong_clf_results = np.empty(n_samples, dtype=np.int8)
        for s in range(n_samples):
            prediction = np.where(
                np.dot(self._strong_classifier_coeffs, classification_results[:, s]) > 0, 1, -1
            ).astype(np.int8)
            strong_clf_results[s] = prediction

        return strong_clf_results

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        '''
        Computes the accuracy of the strong classifier on inputs X against targets y.

        Parameters
        ----------
        X : np.ndarray
            The input data on which to sample the strong classifier.
        y : np.ndarray
            The target classes.

        Returns
        -------
        float
            The accuracy of the model.
        '''

        preds = self.predict(x)
        acc = (preds == y).sum() / len(y)
        return acc

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def __str__(self):
        return f'{self.__class__.__name__}(K={self.K}, B={self.B}, P={self.P}, lbda={self.lbda}, num_reads={self.num_reads})'

    def __repr__(self):
        repr_str = self.__str__()
        clf_limit = 5
        clf_str = (
            f'[{", ".join(map(str, self.classifiers[:clf_limit]))}{", ..." * (len(self.classifiers) > clf_limit)}]'
        )
        repr_str = repr_str[:-2] + f', weak_classifiers={clf_str})'
        return repr_str
