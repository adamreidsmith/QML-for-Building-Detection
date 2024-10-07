import math
from typing import Optional

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def visualize_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    cmap: str = 'viridis',
) -> None:

    geom = o3d.geometry.PointCloud()
    geom.points = o3d.utility.Vector3dVector(points)

    # Use a colormap from Matplotlib
    cmap = plt.get_cmap(cmap)

    # Map integers to colors using the colormap
    colour_var = np.asarray(colors) if colors is not None else points[:, 2]
    colour_var_norm = (colour_var - colour_var.min()) / (colour_var.max() - colour_var.min())
    colors = cmap(colour_var_norm)[:, :3]  # [:3] to exclude the alpha channel
    geom.colors = o3d.utility.Vector3dVector(colors)

    o3d.visualization.draw_geometries([geom])


def confusion_matrix(predictions: np.ndarray, labels: np.ndarray) -> np.ndarray[int]:
    '''
    Compute true positives, true negatives, false positives, and false negatives.

    Parameters
    ----------
    predictions : np.ndarray
        Array of binary predictions
    labels :  np.ndarray
        Array of true binary values

    Returns
    -------
    np.ndarray
        (TP, TN, FP, FN)
    '''

    classes = np.unique(labels)
    assert len(classes) == 2, 'Cannot compute confuction matrix for non-binary classification'

    preds_eq_m1, preds_eq_1 = predictions == -1, predictions == 1
    labels_eq_m1, labels_eq_1 = labels == -1, labels == 1

    # Calculate true positives, true negatives, false positives, and false negatives
    tp = np.sum(preds_eq_1 & labels_eq_1)
    tn = np.sum(preds_eq_m1 & labels_eq_m1)
    fp = np.sum(preds_eq_1 & labels_eq_m1)
    fn = np.sum(preds_eq_m1 & labels_eq_1)

    return np.array((tp, tn, fp, fn), dtype=int)


def f1_score(cm: np.ndarray[int]) -> float:
    '''Compute F1 score from a confusion matrix'''

    tp, _, fp, fn = cm
    numer = 2 * tp
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0.0 else numer / denom


def matthews_corrcoef(cm: np.ndarray[int]) -> float:
    '''Compute Matthew's Correaltion Coefficient from a confusion matrix'''

    tp, tn, fp, fn = cm
    numer = tp * tn - fp * fn
    denom_sqr = (tp + fp) * (tn + fn) * (fp + tn) * (tp + fn)
    return 0.0 if denom_sqr == 0.0 else numer / math.sqrt(denom_sqr)


def accuracy(cm: np.ndarray[int]) -> float:
    '''Compute accuracy from a confusion matrix'''

    tp, tn, fp, fn = cm
    numer = tp + tn
    denom = tp + tn + fp + fn
    return numer / denom
