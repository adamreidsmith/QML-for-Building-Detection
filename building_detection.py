import math
import sys
from pathlib import Path
from typing import Optional, Any
from itertools import product
from collections.abc import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.svm import SVC
from tqdm import tqdm
from dotenv import load_dotenv

from qsvm import QSVM, QSVMGroup
from qboost import QBoost
from adaboost import AdaBoost
from quantum_kernels import (
    Kernel,
    pauli_feature_map,
    iqp_feature_map,
    polynomial_feature_map,
    qaoa_inspired_feature_map,
    data_reuploading_feature_map,
)


# Set DWave API token
load_dotenv()

WORKING_DIR = Path(__file__).parent
LOG_DIR = WORKING_DIR / 'logs'

# SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else np.random.randint(100, 1_000_000)

print(f'Using {SEED = }')

# IID = np.random.randint(1, 1_000_000)
if SEED is not None:
    np.random.seed(SEED)


def visualize_cloud(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
    cmap: str = 'viridis',
    bounds: Optional[tuple[float, float, float, float]] = None,
) -> None:
    if bounds is not None:
        minx, maxx, miny, maxy = bounds
        if minx >= maxx or miny >= maxy:
            raise ValueError('Invalid bounds passed to `visualize_cloud`')

        indices = np.arange(len(points))
        indices = indices[
            (points[:, 0] >= minx) & (points[:, 0] <= maxx) & (points[:, 1] >= miny) & (points[:, 1] <= maxy)
        ]
        points = points[indices]
        colors = colors[indices]

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


def downsample_point_cloud(point_cloud: pd.DataFrame) -> pd.DataFrame:
    '''
    Crop the point cloud to 1/4 of its original area.
    '''

    xmin, xmax, ymin, ymax = point_cloud.x.min(), point_cloud.x.max(), point_cloud.y.min(), point_cloud.y.max()
    xmax = xmin + (xmax - xmin) / 2
    ymax = ymin + (ymax - ymin) / 2

    point_cloud = point_cloud[
        (point_cloud.x >= xmin) & (point_cloud.x <= xmax) & (point_cloud.y >= ymin) & (point_cloud.y <= ymax)
    ]
    point_cloud = point_cloud.reset_index(drop=True)
    return point_cloud


def acc_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    pos_preds = predictions == 1
    neg_preds = predictions == -1
    pos_labels = labels == 1
    neg_labels = labels == -1

    tp = np.sum(pos_preds & pos_labels)
    fp = np.sum(pos_preds & neg_labels)
    fn = np.sum(neg_preds & pos_labels)

    acc = np.sum(predictions == labels) / labels.shape[0]
    f1 = tp / (tp + 0.5 * (fp + fn))

    return acc, f1


def optimize_model(
    model: Any,
    search_space: dict[str, Iterable[Any]],
    kw_params: dict[str, Any],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    x_all: np.ndarray,
    y_all: np.ndarray,
    point_cloud: pd.DataFrame,
    bounds: tuple[float, float, float, float],
    model_name: str = '',
    verbose: bool = True,
    visualize: bool = True,
    fit_takes_valid: bool = False,
    model_kw_params: Optional[dict[str, Any]] = None,
    predict_full_dataset: bool = True,
    score_valid: bool = False,
    write_data: bool = True,
) -> Any:
    if verbose:
        print(f'Optimizing {model_name} model...')

    param_sets = (dict(zip(search_space.keys(), values)) for values in product(*search_space.values()))
    best_clfs = (None, -1.0, [None])  # (classifier, accuracy, [params, ...])
    # Hyperparameter optimization loop
    for params in tqdm(
        param_sets,
        desc=f'Optimizing {model_name}',
        total=math.prod(len(x) for x in search_space.values()),
        disable=not verbose,
    ):
        if model_kw_params is None:
            clf = model(**params, **kw_params)
        else:
            clf = model(params | kw_params, **model_kw_params)  # Occurs for QSVM Group model
        fit_args = (x_train, y_train) if not fit_takes_valid else (x_train, y_train, x_valid, y_valid)
        clf.fit(*fit_args)
        preds = clf.predict(x_valid) if score_valid else clf.predict(x_train)
        acc, f1 = acc_f1(preds, y_valid if score_valid else y_train)
        if acc > best_clfs[1]:
            best_clfs = (clf, acc, [params])
        elif acc == best_clfs[1]:
            best_clfs[2].append(params)
        if write_data:
            with open(LOG_DIR / f'{model_name.lower().replace(" ", "_")}_{SEED}.txt', 'a') as f:
                print(f'{acc:.6f}:{f1:.4f}:{params | kw_params}', file=f)

    clf, acc, param_sets = best_clfs
    if verbose:
        print(f'{model_name} best params with training accuracy {acc:.2%}:')
        for params in param_sets:
            print(f'\t{params | kw_params}')

    train_acc, train_f1 = acc_f1(clf.predict(x_train), y_train)
    print(f'{model_name} training accuracy: {train_acc:.2%}')
    print(f'{model_name} training F1:       {train_f1:.3f}')
    valid_acc, valid_f1 = acc_f1(clf.predict(x_valid), y_valid)
    print(f'{model_name} validation accuracy: {valid_acc:.2%}')
    print(f'{model_name} validation F1:       {valid_f1:.3f}')

    if predict_full_dataset:
        point_cloud_preds = clf.predict(x_all)
        full_acc, full_f1 = acc_f1(point_cloud_preds, y_all)
        print(f'{model_name} full dataset accuracy: {full_acc:.2%}')
        print(f'{model_name} full dataset F1:       {full_f1:.3f}')

        if visualize:
            visualize_cloud(
                point_cloud[['x', 'y', 'z']].to_numpy(),
                colors=point_cloud_preds,
                cmap='cool',
                bounds=bounds,
            )

    return clf


def main():
    data_file = WORKING_DIR / 'data' / '4870E_54560N_kits' / '1m_lidar.csv'
    # data_file = WORKING_DIR / 'data' / '4870E_54560N_kits' / '50cm_lidar.csv'
    full_point_cloud = pd.read_csv(data_file)

    predict_full_dataset = True
    verbose = True
    visualize = True
    num_workers = 6
    write_data = False

    bounds = (full_point_cloud.x.min(), full_point_cloud.x.max(), full_point_cloud.y.min(), full_point_cloud.y.max())

    # Map building points to 1 and all others to 0
    full_point_cloud.classification = full_point_cloud.classification.map(lambda x: 1 if x == 6 else -1)

    if verbose:
        print('Downsampling point cloud...')
    point_cloud = downsample_point_cloud(full_point_cloud)
    if verbose:
        value_counts = point_cloud.classification.value_counts()
        print(
            'Downsampled point cloud:'
            f'\n\tTotal # points:        {len(point_cloud)}'
            f'\n\t# building points:     {value_counts.get(1)}'
            f'\n\t# non-building points: {value_counts.get(-1)}'
        )

    # Choose a random subset of points as a train set
    n_train_samples = 1000
    n_valid_samples = 1000

    indices = np.arange(len(point_cloud))
    np.random.shuffle(indices)
    train_indices = indices[:n_train_samples]
    valid_indices = indices[-n_valid_samples:]

    features = ['z', 'normal_variation', 'height_variation', 'intensity']

    train_x = point_cloud[features].iloc[train_indices].to_numpy()
    valid_x = point_cloud[features].iloc[valid_indices].to_numpy()
    train_y = point_cloud.classification.iloc[train_indices].to_numpy()
    valid_y = point_cloud.classification.iloc[valid_indices].to_numpy()

    train_mean, train_std = np.mean(train_x, axis=0), np.std(train_x, axis=0)
    train_x_normalized = (train_x - train_mean) / train_std
    valid_x_normalized = (valid_x - train_mean) / train_std

    if verbose:
        print(
            'Train set:'
            f'\n\tTotal # points:        {n_train_samples}'
            f'\n\t# building points:     {sum(1 for y in train_y if y == 1)}'
            f'\n\t# non-building points: {sum(1 for y in train_y if y == -1)}'
        )
        print(
            'Validation set:'
            f'\n\tTotal # points:        {n_valid_samples}'
            f'\n\t# building points:     {sum(1 for y in valid_y if y == 1)}'
            f'\n\t# non-building points: {sum(1 for y in valid_y if y == -1)}'
        )

    if visualize:
        visualize_cloud(
            point_cloud[['x', 'y', 'z']].to_numpy(),
            colors=point_cloud.classification.to_numpy(),
            cmap='cool',
            bounds=bounds,
        )

    ###################################################################################################################
    # SVM #############################################################################################################
    ###################################################################################################################

    svm_search_space = {'C': np.geomspace(0.00001, 1000, 25), 'gamma': np.geomspace(0.00001, 100, 22)}
    svm_kw_params = {'class_weight': 'balanced', 'kernel': 'rbf'}

    optimize_model(
        model=SVC,
        search_space=svm_search_space,
        kw_params=svm_kw_params,
        x_train=train_x_normalized,
        y_train=train_y,
        x_valid=valid_x_normalized,
        y_valid=valid_y,
        x_all=(point_cloud[features].to_numpy() - train_mean) / train_std,
        y_all=point_cloud.classification.to_numpy(),
        point_cloud=point_cloud,
        bounds=bounds,
        model_name='SVM',
        verbose=verbose,
        visualize=visualize,
        fit_takes_valid=False,
        model_kw_params=None,
        predict_full_dataset=predict_full_dataset,
        score_valid=False,
        write_data=write_data,
    )

    ###################################################################################################################
    # QSVM ############################################################################################################
    ###################################################################################################################

    qsvm_search_space = {
        'B': [2],
        'P': [0, 1, 2],
        'K': [3, 4, 5, 6],
        'zeta': [0.0, 0.4, 0.8, 1.2],
        'gamma': np.geomspace(0.01 * np.sqrt(10), 10, 6),
    }
    qsvm_kw_params = {'kernel': 'rbf', 'sampler': 'steepest_descent', 'num_reads': 100, 'normalize': True}

    optimize_model(
        model=QSVM,
        search_space=qsvm_search_space,
        kw_params=qsvm_kw_params,
        x_train=train_x,
        y_train=train_y,
        x_valid=valid_x,
        y_valid=valid_y,
        x_all=point_cloud[features].to_numpy(),
        y_all=point_cloud.classification.to_numpy(),
        point_cloud=point_cloud,
        bounds=bounds,
        model_name='QSVM',
        verbose=verbose,
        visualize=visualize,
        fit_takes_valid=False,
        model_kw_params=None,
        predict_full_dataset=predict_full_dataset,
        score_valid=False,
        write_data=write_data,
    )

    ###################################################################################################################
    # QSVM Group ######################################################################################################
    ###################################################################################################################

    # QSVM Group hyperparameters
    S = 20  # Number of classifiers
    M = 50  # Size of subsets
    balance_classes = False

    model_kw_params = {'S': S, 'M': M, 'balance_classes': balance_classes, 'num_workers': num_workers}
    qsvm_group_search_space = {
        'B': [2],
        'P': [0, 1, 2],
        'K': [3, 4, 5, 6],
        'zeta': [0.0, 0.4, 0.8, 1.2],
        'gamma': np.geomspace(0.01 * np.sqrt(10), 10, 6),
    }
    qsvm_group_kw_params = {'kernel': 'rbf', 'sampler': 'steepest_descent', 'num_reads': 100, 'normalize': True}

    optimize_model(
        model=QSVMGroup,
        search_space=qsvm_group_search_space,
        kw_params=qsvm_group_kw_params,
        x_train=train_x,
        y_train=train_y,
        x_valid=valid_x,
        y_valid=valid_y,
        x_all=point_cloud[features].to_numpy(),
        y_all=point_cloud.classification.to_numpy(),
        point_cloud=point_cloud,
        bounds=bounds,
        model_name='QSVM Group',
        verbose=verbose,
        visualize=visualize,
        fit_takes_valid=False,
        model_kw_params=model_kw_params,
        predict_full_dataset=predict_full_dataset,
        score_valid=False,
        write_data=write_data,
    )

    ###################################################################################################################
    # SVM w/ Quantum Kernel ###########################################################################################
    ###################################################################################################################

    n_features = 4

    if verbose:
        print('Defining quantum kernels...')
    uninitialized_kernels = []
    for reps in [1]:
        for entanglement in ['full', 'linear', 'pairwise']:
            for single_pauli in 'XYZ':
                for double_pauli in ['XX', 'YY', 'ZZ']:
                    uninitialized_kernels.append(
                        (
                            pauli_feature_map,
                            dict(
                                num_features=n_features,
                                paulis=[single_pauli, double_pauli],
                                reps=reps,
                                entanglement=entanglement,
                            ),
                        )
                    )
            uninitialized_kernels += [
                (iqp_feature_map, dict(num_features=n_features, reps=reps, entanglement=entanglement)),
                (data_reuploading_feature_map, dict(num_features=n_features, reps=reps, entanglement=entanglement)),
            ]
        uninitialized_kernels += [
            (polynomial_feature_map, dict(num_features=n_features, qubits_per_feature=reps)),
            (qaoa_inspired_feature_map, dict(num_features=n_features, reps=reps)),
        ]

    kernels = []
    for kernel in uninitialized_kernels:
        kernels.append(Kernel(kernel[0](**kernel[1]), f'{kernel[0].__name__}({kernel[1]})'))

    kernel_svm_search_space = {'C': np.geomspace(0.01, 100, 13), 'kernel': kernels}
    kernel_svm_kw_params = {'class_weight': 'balanced'}

    optimize_model(
        model=SVC,
        search_space=kernel_svm_search_space,
        kw_params=kernel_svm_kw_params,
        x_train=train_x_normalized,
        y_train=train_y,
        x_valid=valid_x_normalized,
        y_valid=valid_y,
        x_all=(point_cloud[features].to_numpy() - train_mean) / train_std,
        y_all=point_cloud.classification.to_numpy(),
        point_cloud=point_cloud,
        bounds=bounds,
        model_name='Quantum Kernel SVM',
        verbose=verbose,
        visualize=visualize,
        fit_takes_valid=False,
        model_kw_params=None,
        predict_full_dataset=predict_full_dataset,
        score_valid=False,
        write_data=write_data,
    )

    ###################################################################################################################
    # QSVM w/ Quantum Kernels #########################################################################################
    ###################################################################################################################

    if verbose:
        print('Optimizing QSVM model with Quantum Kernel...')
    kernel_qsvm_search_space = {
        'B': [2],
        'P': [0, 1],
        'K': [3, 4, 5],
        'zeta': [0.0, 0.4, 0.8, 1.2],
        'kernel': kernels,
    }
    kernel_qsvm_kw_params = {'sampler': 'steepest_descent', 'num_reads': 100, 'normalize': True}

    optimize_model(
        model=QSVM,
        search_space=kernel_qsvm_search_space,
        kw_params=kernel_qsvm_kw_params,
        x_train=train_x,
        y_train=train_y,
        x_valid=valid_x,
        y_valid=valid_y,
        x_all=point_cloud[features].to_numpy(),
        y_all=point_cloud.classification.to_numpy(),
        point_cloud=point_cloud,
        bounds=bounds,
        model_name='Quantum Kernel QSVM',
        verbose=verbose,
        visualize=visualize,
        fit_takes_valid=False,
        model_kw_params=None,
        predict_full_dataset=predict_full_dataset,
        score_valid=False,
        write_data=write_data,
    )

    ###################################################################################################################
    # QSVM Group w/ Quantum Kernels ###################################################################################
    ###################################################################################################################

    model_kw_params = {'S': S, 'M': M, 'balance_classes': balance_classes, 'num_workers': num_workers}
    kernel_qsvm_group_search_space = {
        'B': [2],
        'P': [0, 1],
        'K': [3, 4, 5],
        'zeta': [0.0, 0.4, 0.8, 1.2],
        'kernel': kernels,
    }
    kernel_qsvm_group_kw_params = {'sampler': 'steepest_descent', 'num_reads': 100, 'normalize': True}

    optimize_model(
        model=QSVMGroup,
        search_space=kernel_qsvm_group_search_space,
        kw_params=kernel_qsvm_group_kw_params,
        x_train=train_x,
        y_train=train_y,
        x_valid=valid_x,
        y_valid=valid_y,
        x_all=point_cloud[features].to_numpy(),
        y_all=point_cloud.classification.to_numpy(),
        point_cloud=point_cloud,
        bounds=bounds,
        model_name='Quantum Kernel QSVM Group',
        verbose=verbose,
        visualize=visualize,
        fit_takes_valid=False,
        model_kw_params=model_kw_params,
        predict_full_dataset=predict_full_dataset,
        score_valid=False,
        write_data=write_data,
    )

    ###################################################################################################################
    # Weak Classifiers ################################################################################################
    ###################################################################################################################

    n_weak_classifiers = 50
    samples_per_classifier = 20
    balance_classes = True

    # Split the training data into `n_weak_classifiers` random subsets of size `samples_per_classifier`.
    # Use a QSVMGroup to accomplish this.
    qsvm_group = QSVMGroup({}, S=n_weak_classifiers, M=samples_per_classifier, balance_classes=balance_classes)
    qsvm_group._define_data_subsets(train_x, train_y)
    train_x_split = qsvm_group._x_subsets
    train_y_split = qsvm_group._y_subsets

    if verbose:
        print(f'Optimizing {n_weak_classifiers} classifiers with {samples_per_classifier} samples per classifier...')
    # Perform a grid search to optimize each weak classifier
    weak_clf_search_space = {
        'B': [2],
        'P': [0, 1],
        'K': [3, 4, 5],
        'zeta': [0.0, 0.4, 0.8, 1.2],
        'gamma': np.geomspace(0.1, 1, 5),
    }
    weak_clf_kw_params = {'kernel': 'rbf', 'sampler': 'steepest_descent', 'num_reads': 100, 'normalize': True}

    param_sets = [
        dict(zip(weak_clf_search_space.keys(), values)) for values in product(*weak_clf_search_space.values())
    ]
    weak_clf_params = []
    weak_classifiers = []
    for tx, ty in tqdm(zip(train_x_split, train_y_split), desc='Optimizing Weak QSVMs', total=n_weak_classifiers):
        best_weak_qsvm = (None, None, 0.0)
        # Hyperparameter optimization loop
        for params in param_sets:
            qsvm = QSVM(**params, **weak_clf_kw_params)
            qsvm.fit(tx, ty)
            acc = qsvm.score(tx, ty)
            # acc = qsvm.score(train_x, train_y)
            if acc > best_weak_qsvm[2]:
                best_weak_qsvm = (qsvm, params, acc)

        qsvm, params, _ = best_weak_qsvm
        weak_clf_params.append(params)
        weak_classifiers.append(qsvm)

    ###################################################################################################################
    # QBoost ##########################################################################################################
    ###################################################################################################################

    qboost_search_space = {'B': [2], 'P': [0, 1, 2, 3, 4], 'K': [3, 4, 5, 6, 7, 8]}
    qboost_kw_params = {'weak_classifiers': weak_classifiers, 'lbda': (0.0, 2.1, 0.1), 'num_reads': 100}

    optimize_model(
        model=QBoost,
        search_space=qboost_search_space,
        kw_params=qboost_kw_params,
        x_train=train_x,
        y_train=train_y,
        x_valid=valid_x,
        y_valid=valid_y,
        x_all=point_cloud[features].to_numpy(),
        y_all=point_cloud.classification.to_numpy(),
        point_cloud=point_cloud,
        bounds=bounds,
        model_name='QBoost',
        verbose=verbose,
        visualize=visualize,
        fit_takes_valid=True,
        model_kw_params=None,
        predict_full_dataset=predict_full_dataset,
        score_valid=False,
        write_data=write_data,
    )

    ###################################################################################################################
    # AdaBoost ########################################################################################################
    ###################################################################################################################

    if verbose:
        print('Optimizing AdaBoost model...')
    # Run the AdaBoost algorithm on the ensemble of weak QSVMs
    adaboost_search_space = {'n_estimators': list(range(6, 100, 2))}
    adaboost_kw_params = {'weak_classifiers': weak_classifiers}

    optimize_model(
        model=AdaBoost,
        search_space=adaboost_search_space,
        kw_params=adaboost_kw_params,
        x_train=train_x,
        y_train=train_y,
        x_valid=valid_x,
        y_valid=valid_y,
        x_all=point_cloud[features].to_numpy(),
        y_all=point_cloud.classification.to_numpy(),
        point_cloud=point_cloud,
        bounds=bounds,
        model_name='AdaBoost',
        verbose=verbose,
        visualize=visualize,
        fit_takes_valid=False,
        model_kw_params=None,
        predict_full_dataset=predict_full_dataset,
        score_valid=False,
        write_data=write_data,
    )


if __name__ == '__main__':
    main()
