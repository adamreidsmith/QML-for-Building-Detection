import numpy as np


VALID_KERNELS = ['linear', 'poly', 'rbf', 'sigmoid']


def _linear_kernel(x1: np.ndarray, x2: np.ndarray):
    return (x1 * x2).sum(axis=1)


def _norm_sq(x: np.ndarray):
    return _linear_kernel(x, x)


def _create_rbf_kernel(gamma: float):
    def rbf_kernel(x1: np.ndarray, x2: np.ndarray):
        return np.exp(-gamma * _norm_sq(x1 - x2))

    return rbf_kernel


def _create_poly_kernel(gamma: float, coef0: float, degree: int):
    def poly_kernel(x1: np.ndarray, x2: np.ndarray):
        return (gamma * _linear_kernel(x1, x2) + coef0) ** degree

    return poly_kernel


def _create_sig_kernel(gamma: float, coef0: float):
    def sig_kernel(x1: np.ndarray, x2: np.ndarray):
        return np.tanh(gamma * _linear_kernel(x1, x2) + coef0)

    return sig_kernel


def _define_kernel(ker: str, gamma=None, coef0=None, degree=None):
    if ker not in VALID_KERNELS:
        raise ValueError(f'Invalid kernel "{ker}". Valid kernels are {VALID_KERNELS} or a Callable.')
    if ker == 'linear':
        return _linear_kernel
    if ker == 'rbf':
        return _create_rbf_kernel(gamma)
    if ker == 'sigmoid':
        return _create_sig_kernel(gamma, coef0)
    return _create_poly_kernel(gamma, coef0, degree)