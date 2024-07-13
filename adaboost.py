from typing import Callable, Any
import numpy as np


class AdaBoost:
    '''
    AdaBoost classifier using a set of weak classifiers.
    '''

    def __init__(self, weak_classifiers: list[Callable[[np.ndarray], np.ndarray]], n_estimators: int = 50):
        '''
        Initialize the AdaBoost classifier.

        Parameters
        ----------
        weak_classifiers : list[Callable[[np.ndarray], np.ndarray]]
            List of weak classifiers to choose from.
        n_estimators : int
            Number of weak classifiers to use. Default is 50.
        '''

        self.weak_classifiers = weak_classifiers
        self.n_estimators = n_estimators
        self.alphas = []
        self.selected_classifiers = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        '''
        Fit the AdaBoost classifier to the training data.

        Parameters
        ----------
        X : np.ndarray
            Training features of shape (n_samples, n_features)
        y : np.ndarray
            Target labels of shape (n_samples,)
        '''

        n_samples = X.shape[0]
        w = np.ones(n_samples) / n_samples  # Initialize weights uniformly

        for _ in range(self.n_estimators):
            # Calculate error for each weak classifier
            errors = np.array([np.sum(w[y != clf(X)]) for clf in self.weak_classifiers])

            # Select the best weak classifier
            best_clf_idx = np.argmin(errors)
            best_clf = self.weak_classifiers[best_clf_idx]

            # Calculate classifier weight
            error = errors[best_clf_idx]
            if error >= 1.0:
                break
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))

            # Update sample weights
            predictions = best_clf(X)
            w *= np.exp(-alpha * y * predictions)
            w /= w.sum()  # Normalize weights

            # Store the selected classifier and its weight
            self.alphas.append(alpha)
            self.selected_classifiers.append(best_clf)

    def predict(self, X: np.ndarray) -> np.ndarray:
        '''
        Make predictions using the trained AdaBoost classifier.

        Parameters
        ----------
        X : np.ndarray
            Input features of shape (n_samples, n_features)

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples,)
        '''

        clf_preds = np.array([clf(X) for clf in self.selected_classifiers])
        return np.sign(np.dot(self.alphas, clf_preds))

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        '''
        Compute the accuracy of the GBM on inputs X against targets y.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features)
        y : np.ndarray
            Target vector of shape (n_samples,)

        Returns
        -------
        float
            The accuracy of the GBM.
        '''

        preds = self.predict(X)
        acc = np.mean(preds == y)
        return acc

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)
