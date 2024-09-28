import math
import sys
from pathlib import Path
from typing import Optional, Any
from itertools import product
from collections.abc import Callable, Iterable

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
):
    if verbose:
        print(f'Optimizing {model_name} model...')

    param_sets = (dict(zip(search_space.keys(), values)) for values in product(*search_space.values()))
    best_clf = (None, None, 0.0)
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
        acc = clf.score(x_train, y_train)
        # acc = clf.score(x_valid, y_valid)
        if acc > best_clf[2]:
            best_clf = (clf, params, acc)
        with open(LOG_DIR / f'{model_name.lower().replace(" ", "_")}_{SEED}.txt', 'a') as f:
            print(f'{acc:.5f}:{params | kw_params}', file=f)

    clf, params, _ = best_clf
    if verbose:
        print(f'{model_name} params: {params | kw_params}')

    train_acc = clf.score(x_train, y_train)
    print(f'{model_name} training accuracy: {train_acc:.2%}')
    valid_acc = clf.score(x_valid, y_valid)
    print(f'{model_name} validation accuracy: {valid_acc:.2%}')
    point_cloud_preds = clf.predict(x_all)
    full_acc = (point_cloud_preds == y_all).sum() / len(x_all)
    print(f'{model_name} full dataset accuracy: {full_acc:.2%}')

    if visualize:
        visualize_cloud(
            point_cloud[['x', 'y', 'z']].to_numpy(),
            colors=point_cloud_preds,
            cmap='cool',
            bounds=bounds,
        )


def main(verbose: bool = True, visualize: bool = True, num_workers: int = 5):
    data_file = WORKING_DIR / 'data' / '4870E_54560N_kits' / '1m_lidar.csv'
    # data_file = WORKING_DIR / 'data' / '4870E_54560N_kits' / '50cm_lidar.csv'
    full_point_cloud = pd.read_csv(data_file)

    bounds = (full_point_cloud.x.min(), full_point_cloud.x.max(), full_point_cloud.y.min(), full_point_cloud.y.max())
    # visualize_cloud(
    #     full_point_cloud[['x', 'y', 'z']].to_numpy(), colors=full_point_cloud.classification.to_numpy(), bounds=bounds
    # )

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
    )

    # param_sets = (dict(zip(svm_search_space.keys(), values)) for values in product(*svm_search_space.values()))
    # best_svm = (None, None, 0.0)
    # # Hyperparameter optimization loop
    # for params in tqdm(param_sets, desc='Optimizing SVM', total=math.prod(len(x) for x in svm_search_space.values())):
    #     svm = SVC(**params, **svm_kw_params)
    #     svm.fit(train_x_normalized, train_y)
    #     acc = svm.score(train_x_normalized, train_y)
    #     # acc = svm.score(valid_x_normalized, valid_y)
    #     if acc > best_svm[2]:
    #         best_svm = (svm, params, acc)
    #     with open(LOG_DIR / f'svm_{SEED}.txt', 'a') as f:
    #         print(f'{acc:.5f}:{params | svm_kw_params}', file=f)

    # svm, params, _ = best_svm
    # if verbose:
    #     print(f'SVM params: {params | svm_kw_params}')

    # svm_train_acc = svm.score(train_x_normalized, train_y)
    # print(f'SVM training accuracy: {svm_train_acc:.2%}')
    # svm_valid_acc = svm.score(valid_x_normalized, valid_y)
    # print(f'SVM validation accuracy: {svm_valid_acc:.2%}')
    # point_cloud_preds = svm.predict((point_cloud[features].to_numpy() - train_mean) / train_std)
    # svm_acc = (point_cloud_preds == point_cloud.classification.to_numpy()).sum() / len(point_cloud)
    # print(f'SVM full dataset accuracy: {svm_acc:.2%}')

    # if visualize:
    #     visualize_cloud(
    #         point_cloud[['x', 'y', 'z']].to_numpy(),
    #         colors=point_cloud_preds,
    #         cmap='cool',
    #         bounds=bounds,
    #     )

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
    )

    # param_sets = (dict(zip(qsvm_search_space.keys(), values)) for values in product(*qsvm_search_space.values()))
    # best_qsvm = (None, None, 0.0)
    # # Hyperparameter optimization loop
    # for params in tqdm(
    #     param_sets, desc='Optimizing QSVM', total=math.prod(len(x) for x in qsvm_search_space.values())
    # ):
    #     qsvm = QSVM(**params, **qsvm_kw_params)
    #     qsvm.fit(train_x, train_y)
    #     acc = qsvm.score(train_x, train_y)
    #     # acc = qsvm.score(valid_x, valid_y)
    #     if acc > best_qsvm[2]:
    #         best_qsvm = (qsvm, params, acc)
    #     with open(LOG_DIR / f'qsvm_{SEED}.txt', 'a') as f:
    #         print(f'{acc:.5f}:{params | qsvm_kw_params}', file=f)

    # qsvm, params, _ = best_qsvm  # The best QSVM
    # if verbose:
    #     print(f'QSVM params: {params | qsvm_kw_params}')

    # qsvm_train_acc = qsvm.score(train_x, train_y)
    # print(f'QSVM training accuracy: {qsvm_train_acc:.2%}')
    # qsvm_valid_acc = qsvm.score(valid_x, valid_y)
    # print(f'QSVM validation accuracy: {qsvm_valid_acc:.2%}')
    # point_cloud_preds = qsvm.predict(point_cloud[features].to_numpy())
    # qsvm_acc = (point_cloud_preds == point_cloud.classification.to_numpy()).sum() / len(point_cloud)
    # print(f'QSVM full dataset accuracy: {qsvm_acc:.2%}')

    # if visualize:
    #     visualize_cloud(
    #         point_cloud[['x', 'y', 'z']].to_numpy(),
    #         colors=point_cloud_preds,
    #         cmap='cool',
    #         bounds=bounds,
    #     )

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
    )

    # param_sets = (
    #     dict(zip(qsvm_group_search_space.keys(), values)) for values in product(*qsvm_group_search_space.values())
    # )
    # best_qsvm_group = (None, None, 0.0)
    # # Hyperparameter optimization loop
    # for params in tqdm(
    #     param_sets, desc='Optimizing QSVM Group', total=math.prod(len(x) for x in qsvm_group_search_space.values())
    # ):
    #     qsvm_group = QSVMGroup(
    #         params | qsvm_group_kw_params, S=S, M=M, balance_classes=balance_classes, num_workers=num_workers
    #     )
    #     qsvm_group.fit(train_x, train_y)
    #     acc = qsvm_group.score(train_x, train_y)
    #     # acc = qsvm_group.score(valid_x, valid_y)
    #     if acc > best_qsvm_group[2]:
    #         best_qsvm_group = (qsvm_group, params, acc)
    #     with open(LOG_DIR / f'qsvm_group_{SEED}.txt', 'a') as f:
    #         print(f'{acc:.5f}:{params | qsvm_group_kw_params}', file=f)

    # qsvm_group, params, _ = best_qsvm_group  # The best QSVM Group
    # if verbose:
    #     print(f'QSVM Group params: {params | qsvm_group_kw_params}')

    # qsvm_group_train_acc = qsvm_group.score(train_x, train_y)
    # print(f'QSVM Group training accuracy: {qsvm_group_train_acc:.2%}')
    # qsvm_group_valid_acc = qsvm_group.score(valid_x, valid_y)
    # print(f'QSVM Group validation accuracy: {qsvm_group_valid_acc:.2%}')
    # point_cloud_preds = qsvm_group.predict(point_cloud[features].to_numpy())
    # qsvm_group_acc = (point_cloud_preds == point_cloud.classification.to_numpy()).sum() / len(point_cloud)
    # print(f'QSVM Group full dataset accuracy: {qsvm_group_acc:.2%}')

    # if visualize:
    #     visualize_cloud(
    #         point_cloud[['x', 'y', 'z']].to_numpy(),
    #         colors=point_cloud_preds,
    #         cmap='cool',
    #         bounds=bounds,
    #     )

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
    )

    # param_sets = (
    #     dict(zip(kernel_svm_search_space.keys(), values)) for values in product(*kernel_svm_search_space.values())
    # )
    # best_kernel_svm = (None, None, 0.0)
    # # Hyperparameter optimization loop
    # for params in tqdm(
    #     param_sets, desc='Optimizing Kernel SVM', total=math.prod(len(x) for x in kernel_svm_search_space.values())
    # ):
    #     kernel_svm = SVC(**params, **kernel_svm_kw_params)
    #     kernel_svm.fit(train_x_normalized, train_y)
    #     acc = kernel_svm.score(train_x_normalized, train_y)
    #     # acc = kernel_svm.score(valid_x, valid_y)
    #     if acc > best_kernel_svm[2]:
    #         best_kernel_svm = (kernel_svm, params, acc)
    #     with open(LOG_DIR / f'kernel_svm_{SEED}.txt', 'a') as f:
    #         print(f'{acc:.5f}:{params | kernel_svm_kw_params}', file=f)

    # kernel_svm, params, _ = best_kernel_svm  # The best Quantum Kernel SVM
    # if verbose:
    #     print(f'Quantum Kernel SVM params: {params | kernel_svm_kw_params}')

    # kernel_svm_train_acc = kernel_svm.score(train_x_normalized, train_y)
    # print(f'Quantum Kernel SVM training accuracy: {kernel_svm_train_acc:.2%}')
    # kernel_svm_valid_acc = kernel_svm.score(valid_x_normalized, valid_y)
    # print(f'Quantum Kernel SVM validation accuracy: {kernel_svm_valid_acc:.2%}')
    # point_cloud_preds = kernel_svm.predict((point_cloud[features].to_numpy() - train_mean) / train_std)
    # kernel_svm_acc = (point_cloud_preds == point_cloud.classification.to_numpy()).sum() / len(point_cloud)
    # print(f'Quantum Kernel SVM full dataset accuracy: {kernel_svm_acc:.2%}')

    # if visualize:
    #     visualize_cloud(
    #         point_cloud[['x', 'y', 'z']].to_numpy(),
    #         colors=point_cloud_preds,
    #         cmap='cool',
    #         bounds=bounds,
    #     )

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
    )

    # param_sets = (
    #     dict(zip(kernel_qsvm_search_space.keys(), values)) for values in product(*kernel_qsvm_search_space.values())
    # )
    # best_kernel_qsvm = (None, None, 0.0)
    # # Hyperparameter optimization loop
    # for params in tqdm(
    #     param_sets,
    #     desc='Optimizing Quantum Kernel QSVM',
    #     total=math.prod(len(x) for x in kernel_qsvm_search_space.values()),
    # ):
    #     kernel_qsvm = QSVM(**params, **kernel_qsvm_kw_params)
    #     kernel_qsvm.fit(train_x, train_y)
    #     acc = kernel_qsvm.score(train_x, train_y)
    #     # acc = kernel_qsvm.score(valid_x, valid_y)
    #     if acc > best_kernel_qsvm[2]:
    #         best_kernel_qsvm = (kernel_qsvm, params, acc)
    #     with open(LOG_DIR / f'kernel_qsvm_{SEED}.txt', 'a') as f:
    #         print(f'{acc:.5f}:{params | kernel_qsvm_kw_params}', file=f)

    # kernel_qsvm, params, _ = best_kernel_qsvm  # The best Quantum Kernel QSVM
    # if verbose:
    #     print(f'Quantum Kernel QSVM params: {params | kernel_qsvm_kw_params}')

    # kernel_qsvm_train_acc = kernel_qsvm.score(train_x, train_y)
    # print(f'Quantum Kernel QSVM training accuracy: {kernel_qsvm_train_acc:.2%}')
    # kernel_qsvm_valid_acc = kernel_qsvm.score(valid_x, valid_y)
    # print(f'Quantum Kernel QSVM validation accuracy: {kernel_qsvm_valid_acc:.2%}')
    # point_cloud_preds = kernel_qsvm.predict(point_cloud[features].to_numpy())
    # kernel_qsvm_acc = (point_cloud_preds == point_cloud.classification.to_numpy()).sum() / len(point_cloud)
    # print(f'Quantum Kernel QSVM full dataset accuracy: {kernel_qsvm_acc:.2%}')

    # if visualize:
    #     visualize_cloud(
    #         point_cloud[['x', 'y', 'z']].to_numpy(),
    #         colors=point_cloud_preds,
    #         cmap='cool',
    #         bounds=bounds,
    #     )

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
    )

    # param_sets = (
    #     dict(zip(kernel_qsvm_group_search_space.keys(), values))
    #     for values in product(*kernel_qsvm_group_search_space.values())
    # )
    # best_kernel_qsvm_group = (None, None, 0.0)
    # # Hyperparameter optimization loop
    # for params in tqdm(
    #     param_sets,
    #     desc='Optimizing Quantum Kernel QSVM Group',
    #     total=math.prod(len(x) for x in kernel_qsvm_group_search_space.values()),
    # ):
    #     kernel_qsvm_group = QSVMGroup(
    #         params | kernel_qsvm_group_kw_params, S=S, M=M, balance_classes=balance_classes, num_workers=num_workers
    #     )
    #     kernel_qsvm_group.fit(train_x, train_y)
    #     acc = kernel_qsvm_group.score(train_x, train_y)
    #     # acc = kernel_qsvm_group.score(valid_x, valid_y)
    #     if acc > best_kernel_qsvm_group[2]:
    #         best_kernel_qsvm_group = (kernel_qsvm_group, params, acc)
    #     with open(LOG_DIR / f'kernel_qsvm_group_{SEED}.txt', 'a') as f:
    #         print(f'{acc:.5f}:{params | kernel_qsvm_group_kw_params}', file=f)

    # kernel_qsvm_group, params, _ = best_kernel_qsvm_group  # The best Quantum Kernel QSVM Group
    # if verbose:
    #     print(f'Quantum Kernel QSVM Group params: {params | kernel_qsvm_group_kw_params}')

    # kernel_qsvm_group_train_acc = kernel_qsvm_group.score(train_x, train_y)
    # print(f'Quantum Kernel QSVM Group training accuracy: {kernel_qsvm_group_train_acc:.2%}')
    # kernel_qsvm_group_valid_acc = kernel_qsvm_group.score(valid_x, valid_y)
    # print(f'Quantum Kernel QSVM Group validation accuracy: {kernel_qsvm_group_valid_acc:.2%}')
    # point_cloud_preds = kernel_qsvm_group.predict(point_cloud[features].to_numpy())
    # kernel_qsvm_group_acc = (point_cloud_preds == point_cloud.classification.to_numpy()).sum() / len(point_cloud)
    # print(f'Quantum Kernel QSVM Group full dataset accuracy: {kernel_qsvm_group_acc:.2%}')

    # if visualize:
    #     visualize_cloud(
    #         point_cloud[['x', 'y', 'z']].to_numpy(),
    #         colors=point_cloud_preds,
    #         cmap='cool',
    #         bounds=bounds,
    #     )

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
    )

    # param_sets = [dict(zip(qboost_search_space.keys(), values)) for values in product(*qboost_search_space.values())]
    # best_qboost = (None, None, 0.0)
    # # Hyperparameter optimization loop
    # for params in tqdm(
    #     param_sets, desc='Optimizing QBoost', total=math.prod(len(x) for x in qboost_search_space.values())
    # ):
    #     qboost = QBoost(**params, **qboost_kw_params)
    #     qboost.fit(train_x, train_y, valid_x, valid_y)
    #     acc = qboost.score(train_x, train_y)
    #     # acc = qboost.score(valid_x, valid_y)
    #     if acc > best_qboost[2]:
    #         best_qboost = (qboost, params, acc)
    #     with open(LOG_DIR / f'qboost_{SEED}.txt', 'a') as f:
    #         print(f'{acc:.5f}:{params | qboost_kw_params}', file=f)

    # qboost, params, _ = best_qboost  # The best QBoost
    # if verbose:
    #     print(f'QBoost params: {params | qboost_kw_params}')

    # qboost_train_acc = qboost.score(train_x, train_y)
    # print(f'QBoost training accuracy: {qboost_train_acc:.2%}')
    # qboost_valid_acc = qboost.score(valid_x, valid_y)
    # print(f'QBoost validation accuracy: {qboost_valid_acc:.2%}')
    # point_cloud_preds = qboost.predict(point_cloud[features].to_numpy())
    # qboost_acc = (point_cloud_preds == point_cloud.classification.to_numpy()).sum() / len(point_cloud)
    # print(f'QBoost full dataset accuracy: {qboost_acc:.2%}')

    # if visualize:
    #     visualize_cloud(
    #         point_cloud[['x', 'y', 'z']].to_numpy(),
    #         colors=point_cloud_preds,
    #         cmap='cool',
    #         bounds=bounds,
    #     )

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
    )

    # param_sets = [
    #     dict(zip(adaboost_search_space.keys(), values)) for values in product(*adaboost_search_space.values())
    # ]
    # best_adaboost = (None, None, 0.0)
    # # Hyperparameter optimization loop
    # for params in tqdm(
    #     param_sets, desc='Optimizing AdaBoost', total=math.prod(len(x) for x in adaboost_search_space.values())
    # ):
    #     adaboost = AdaBoost(**params, **adaboost_kw_params)
    #     adaboost.fit(train_x, train_y)
    #     acc = adaboost.score(train_x, train_y)
    #     # acc = adaboost.score(valid_x, valid_y)
    #     if acc > best_adaboost[2]:
    #         best_adaboost = (adaboost, params, acc)
    #     with open(LOG_DIR / f'adaboost_{SEED}.txt', 'a') as f:
    #         print(f'{acc:.5f}:{params | adaboost_kw_params}', file=f)

    # adaboost, params, _ = best_adaboost  # The best AdaBoost
    # if verbose:
    #     print(f'AdaBoost params: {params | adaboost_kw_params}')

    # adaboost_train_acc = adaboost.score(train_x, train_y)
    # print(f'AdaBoost training accuracy: {adaboost_train_acc:.2%}')
    # adaboost_valid_acc = adaboost.score(valid_x, valid_y)
    # print(f'AdaBoost validation accuracy: {adaboost_valid_acc:.2%}')
    # point_cloud_preds = adaboost.predict(point_cloud[features].to_numpy())
    # adaboost_acc = (point_cloud_preds == point_cloud.classification.to_numpy()).sum() / len(point_cloud)
    # print(f'AdaBoost full dataset accuracy: {adaboost_acc:.2%}')

    # if visualize:
    #     visualize_cloud(
    #         point_cloud[['x', 'y', 'z']].to_numpy(),
    #         colors=point_cloud_preds,
    #         cmap='cool',
    #         bounds=bounds,
    #     )


if __name__ == '__main__':
    main(visualize=False)
