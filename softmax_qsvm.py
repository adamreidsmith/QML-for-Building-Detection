from typing import Any, Optional
from collections.abc import Sequence, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import softmax

from qsvm import QSVM
from utils import confusion_matrix, matthews_corrcoef


class SoftmaxQSVM:
    def __init__(
        self,
        weak_classifiers: Optional[Sequence[Callable[[np.ndarray], np.ndarray]]] = None,
        qsvm_params: Optional[dict[str, Any]] = None,
        S: int = 50,
        M: int = 20,
        multiplier: float = 1.0,
        balance_classes: bool = True,
        num_workers: int = 1,
    ) -> None:
        '''
        Parameters
        ----------
        weak_classifiers : Sequence[Callable[[np.ndarray], np.ndarray]] | None
            A list of classifiers on which to train the model. If None, qsvm_params must be provided so that
            weak classifiers can be generated. Default is None.
        qsvm_params : dict[str, Any] | None
            A dictionary of keyword parameters to initialize each QSVM. Only applies if weak_classifiers is
            None. Default is None.
        S : int, optional
            The number of classifiers to consider. Default is 50.
        M : int, optional
            The size of the subsets each QSVM is trained on. Default is 20.
        balance_classes : bool, optional
            Whether or not to balance the number of points from each class in every M-sample subset. Default
            is True.
        num_workers : int, optional
            The number of threads to launch for parallelization. Default is 1.
        '''

        if weak_classifiers is not None and qsvm_params is not None:
            raise ValueError('Supply exactly one of `weak_classifiers` or `qsvm_params`, not both')
        if weak_classifiers is None and qsvm_params is None:
            raise ValueError('Exactly one of `weak_classifiers` or `qsvm_params`, must be supplied')

        self.weak_classifiers = weak_classifiers
        if self.weak_classifiers is not None:
            self.weak_classifiers = list(self.weak_classifiers)

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

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SoftmaxQSVM':
        '''
        Fit the SoftmaxQSVM on input data X and corresponding labels y.

        Parameters
        ----------
        X : np.ndarray
            An array of shape (n_samples, n_features) of training data.
        y : np.ndarray
            An array of shape (n_samples,) of binary class labels for the training data.
        '''

        if self.weak_classifiers is None:
            self._define_data_subsets(X, y)

            def train_and_score(subsets: tuple[np.ndarray, np.ndarray]) -> tuple[QSVM, float]:
                x_subset, y_subset = subsets
                qsvm = QSVM(**self.qsvm_params)
                qsvm.fit(x_subset, y_subset)
                preds = qsvm.predict(X)
                cm = confusion_matrix(preds, y)
                mcc = matthews_corrcoef(cm)
                return qsvm, mcc

            # Use ThreadPoolExecutor to parallelize the process
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                results = list(executor.map(train_and_score, zip(self._x_subsets, self._y_subsets)))

            # Unpack the results into _trained_qsvms and accuracies
            self._trained_qsvms, mccs = zip(*results)
            mccs = np.asarray(mccs)
        else:
            mccs = np.fromiter(
                (matthews_corrcoef(confusion_matrix(clf.predict(X), y)) for clf in self.weak_classifiers),
                dtype=float,
            )

        self._weights = softmax(self.multiplier * mccs).reshape(-1, 1)

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
        classifiers = self.weak_classifiers if self.weak_classifiers is not None else self._trained_qsvms
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            results = list(executor.map(lambda clf: clf.predict(X), classifiers))
        results = np.asarray(results)

        # Combine predictions as a weighted sum
        preds = (self._weights * results).sum(axis=0)
        preds = np.where(preds > 0, 1, -1)

        return preds

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        '''
        Computes the accuracy of the SoftmaxQSVM on inputs X against targets y.

        Parameters
        ----------
        X : np.ndarray
            The input data on which to run the QSVM.
        y : np.ndarray
            The target classes.

        Returns
        -------
        float
            The accuracy of the SoftmaxQSVM.
        '''

        y = y.astype(int)

        preds = self.predict(X)  # Normalization happens here
        acc = (preds == y).sum() / len(y)
        return acc
