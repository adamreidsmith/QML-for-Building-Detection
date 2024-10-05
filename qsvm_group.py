from typing import Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.special import softmax

from qsvm import QSVM


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
        preds = np.where(preds > 0, 1, -1)

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

        y = y.astype(int)

        preds = self.predict(X)  # Normalization happens here
        acc = (preds == y).sum() / len(y)
        return acc
