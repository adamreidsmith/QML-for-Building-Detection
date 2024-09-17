import operator
import random
import math
import sys
from pathlib import Path
from typing import Optional, Any
from collections.abc import Sequence
from itertools import product
from multiprocessing import Pool
from functools import partial

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.svm import SVC
from tqdm import tqdm
from dotenv import load_dotenv

from qsvm import QSVM
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
PROCESSES = 1
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 40

IID = np.random.randint(1, 1_000_000)
random.seed(SEED)
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


def gs_helper(param_set, svm_algo, svm_kw_params, train_x, train_y, valid_x, valid_y):
    '''
    Helper function for the grid search function to facilitate multiprocessing.
    '''

    svm = svm_algo(**svm_kw_params, **param_set)
    if isinstance(svm, QBoost):
        svm.fit(train_x, train_y, valid_x, valid_y)
    else:
        svm.fit(train_x, train_y)
    acc = svm.score(valid_x, valid_y)
    return param_set, acc


def grid_search(
    svm_algo: QSVM | SVC | QBoost,
    search_space: dict[str, Sequence[Any]],
    train_x: np.ndarray,
    train_y: np.ndarray,
    valid_x: Optional[np.ndarray] = None,
    valid_y: Optional[np.ndarray] = None,
    svm_kw_params: dict[str, Any] = {},
    processes: int = 1,
    save_path: Optional[Path | str] = None,
) -> tuple[dict[str, Any], float]:
    '''
    Perform grid search on the QSVM, SVC, or QBoost algorithm.
    '''

    # Use the training data as the validation data if validation data is not supplied
    if valid_x is None:
        valid_x = train_x
        valid_y = train_y

    train_svm_partial = partial(
        gs_helper,
        svm_algo=svm_algo,
        svm_kw_params=svm_kw_params,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
    )

    # Loop over the search space and train every model
    params, param_values = zip(*search_space.items())
    param_sets = (dict(zip(params, param_vals)) for param_vals in product(*param_values))
    if processes == 1:
        results = []
        for param_set in tqdm(param_sets, total=math.prod(len(x) for x in param_values)):
            results.append(train_svm_partial(param_set))
    else:
        with Pool(processes) as pool:
            results = pool.map(train_svm_partial, param_sets, chunksize=1)

    results = sorted(results, key=operator.itemgetter(1), reverse=True)

    if isinstance(save_path, (Path, str)):
        results_str = (str(r) + '\n' for r in results)
        with open(save_path, 'w') as f:
            f.writelines(results_str)

    return results[0]


def pfm_preprocessing(x: np.ndarray) -> np.ndarray:
    '''
    Preprocessing function for the polynomial feature map.
    '''

    return np.arcsin((x + 1) % 2 - 1)


def main(verbose: bool = True, visualize: bool = True):
    data_file = WORKING_DIR / 'data' / '1m_lidar.csv'
    # data_file = WORKING_DIR / 'data' / '50cm_lidar.csv'
    full_point_cloud = pd.read_csv(data_file)

    bounds = (487040.0, 487330.0, 5456140.0, 5456340.0)
    # visualize_cloud(
    #     full_point_cloud[['x', 'y', 'z']].to_numpy(), colors=full_point_cloud.classification.to_numpy(), bounds=bounds
    # )

    # classification value counts
    # 5 (high vegetation)    1806128
    # 6 (building)           716277
    # 2 (ground)             705938
    # 3 (low vegetation)     233988
    full_point_cloud.classification = full_point_cloud.classification.map(lambda x: 1 if x == 6 else -1)

    if verbose:
        print('Downsampling point cloud...')
    point_cloud = downsample_point_cloud(full_point_cloud)
    if verbose:
        value_counts = point_cloud.classification.value_counts()
        print('Downsampled point cloud:')
        print(f'    Total # points:        {len(point_cloud)}')
        print(f'    # building points:     {value_counts.get(1)}')
        print(f'    # non-building points: {value_counts.get(-1)}')

    # Choose a random subset of points as a train set
    n_train_samples = 100
    n_valid_samples = 100

    indices = np.arange(len(point_cloud))
    random.shuffle(indices)
    train_indices = indices[:n_train_samples]
    valid_indices = indices[-n_valid_samples:]
    # train_indices_c1 = indices[point_cloud.classification[indices] == -1][: 1 * n_train_samples // 2]
    # train_indices_c2 = indices[point_cloud.classification[indices] == 1][: 1 * n_train_samples // 2]
    # train_indices = np.hstack((train_indices_c1, train_indices_c2))
    # random.shuffle(train_indices)
    # valid_indices_c1 = indices[point_cloud.classification[indices] == -1][-1 * n_valid_samples // 2 :]
    # valid_indices_c2 = indices[point_cloud.classification[indices] == 1][-1 * n_valid_samples // 2 :]
    # valid_indices = np.hstack((valid_indices_c1, valid_indices_c2))
    # random.shuffle(valid_indices)

    features = ['z', 'normal_variation', 'height_variation', 'intensity']

    train_x = point_cloud[features].iloc[train_indices].to_numpy()
    valid_x = point_cloud[features].iloc[valid_indices].to_numpy()
    train_y = point_cloud.classification.iloc[train_indices].to_numpy()
    valid_y = point_cloud.classification.iloc[valid_indices].to_numpy()

    if verbose:
        print('Train set:')
        print(f'    Total # points:        {n_train_samples}')
        print(f'    # building points:     {sum(1 for y in train_y if y == 1)}')
        print(f'    # non-building points: {sum(1 for y in train_y if y == -1)}')
        print('Validation set:')
        print(f'    Total # points:        {n_valid_samples}')
        print(f'    # building points:     {sum(1 for y in valid_y if y == 1)}')
        print(f'    # non-building points: {sum(1 for y in valid_y if y == -1)}')

    if visualize:
        visualize_cloud(
            point_cloud[['x', 'y', 'z']].to_numpy(),
            colors=point_cloud.classification.to_numpy(),
            cmap='cool',
            bounds=bounds,
        )

    train_mean, train_std = np.mean(train_x, axis=0), np.std(train_x, axis=0)

    ###################################################################################################################
    # SVM #############################################################################################################
    ###################################################################################################################

    if verbose:
        print('Optimizing SVM model...')
    svm_search_space = {'C': np.geomspace(0.00001, 1000, 25), 'gamma': np.geomspace(0.00001, 100, 22)}
    svm_kw_params = {'class_weight': 'balanced', 'kernel': 'rbf'}
    params, _ = grid_search(
        SVC,
        search_space=svm_search_space,
        svm_kw_params=svm_kw_params,
        train_x=(train_x - train_mean) / train_std,
        train_y=train_y,
        valid_x=(valid_x - train_mean) / train_std,
        valid_y=valid_y,
        processes=PROCESSES,
    )
    if verbose:
        print(f'SVM params: {params | svm_kw_params}')
    svm = SVC(**params, **svm_kw_params)
    svm.fit((train_x - train_mean) / train_std, train_y)

    svm_acc = svm.score((valid_x - train_mean) / train_std, valid_y)
    print(f'SVM validation accuracy: {svm_acc:.2%}')

    svm_acc = svm.score(
        (point_cloud[features].to_numpy() - train_mean) / train_std, point_cloud.classification.to_numpy()
    )
    print(f'SVM accuracy: {svm_acc:.2%}')

    if visualize:
        visualize_cloud(
            point_cloud[['x', 'y', 'z']].to_numpy(),
            colors=svm.predict(point_cloud[features].to_numpy()),
            cmap='cool',
            bounds=bounds,
        )

    ###################################################################################################################
    # QSVM ############################################################################################################
    ###################################################################################################################

    if verbose:
        print('Optimizing QSVM model...')
    qsvm_search_space = {
        'B': [2],
        'P': [0, 1, 2],
        'K': [2, 3, 4, 5, 6],
        'zeta': [0.5, 1, 1.5, 2],
        'gamma': np.geomspace(0.01 * np.sqrt(10), 10, 6),
    }
    qsvm_kw_params = {'kernel': 'rbf', 'sampler': 'steepest_descent', 'num_reads': 100, 'normalize': True}
    params, _ = grid_search(
        QSVM,
        search_space=qsvm_search_space,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        processes=PROCESSES,
        svm_kw_params=qsvm_kw_params,
    )
    if verbose:
        print(f'QSVM params: {params | qsvm_kw_params}')
    qsvm = QSVM(**params, **qsvm_kw_params)
    qsvm.fit(train_x, train_y)

    qsvm_acc = qsvm.score(valid_x, valid_y)
    print(f'QSVM validation accuracy: {qsvm_acc:.2%}')

    qsvm_acc = qsvm.score(point_cloud[features].to_numpy(), point_cloud.classification.to_numpy())
    print(f'QSVM accuracy: {qsvm_acc:.2%}')

    if visualize:
        visualize_cloud(
            point_cloud[['x', 'y', 'z']].to_numpy(),
            colors=qsvm(point_cloud[features].to_numpy()),
            cmap='cool',
            bounds=bounds,
        )

    ###################################################################################################################
    # SVM w/ Quantum Kernel ###########################################################################################
    ###################################################################################################################

    n_features = 4

    if verbose:
        print('Defining quantum kernels...')
    uninitialized_kernels = []
    # for reps in [1, 2]:
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
                            {},
                        )
                    )
            uninitialized_kernels += [
                (iqp_feature_map, dict(num_features=n_features, reps=reps, entanglement=entanglement), {}),
                (
                    data_reuploading_feature_map,
                    dict(num_features=n_features, reps=reps, entanglement=entanglement),
                    {},
                ),
            ]
        uninitialized_kernels += [
            (
                polynomial_feature_map,
                dict(num_features=n_features, qubits_per_feature=reps),
                {'preprocess_func': pfm_preprocessing},
            ),
            (qaoa_inspired_feature_map, dict(num_features=n_features, reps=reps), {}),
        ]

    kernels = []
    for kernel in uninitialized_kernels:
        kernels.append(Kernel(kernel[0](**kernel[1]), f'{kernel[0].__name__}({kernel[1]})', **kernel[2]))

    if verbose:
        print('Optimizing SVM model with Quantum Kernel...')
    svm_search_space = {'C': np.geomspace(0.01, 10, 10), 'kernel': kernels}
    svm_kw_params = {'class_weight': 'balanced'}
    params, _ = grid_search(
        SVC,
        search_space=svm_search_space,
        svm_kw_params=svm_kw_params,
        train_x=(train_x - train_mean) / train_std,
        train_y=train_y,
        valid_x=(valid_x - train_mean) / train_std,
        valid_y=valid_y,
        processes=1,  # `qiskit_machine_learning.kernels.FidelityStatevectorKernel` is not picklable
        # save_path=WORKING_DIR / 'results2' / f'{SEED}_SVC_results_{IID}.txt',
        save_path=None,
    )
    if verbose:
        print(f'SVM params: {params | svm_kw_params}')
    svm = SVC(**params, **svm_kw_params)
    svm.fit((train_x - train_mean) / train_std, train_y)

    svm_acc = svm.score((valid_x - train_mean) / train_std, valid_y)
    print(f'SVM validation accuracy: {svm_acc:.2%}')

    # Predicting with a quantum kernel is expensive due to the number of circuits that must be simulated,
    # so we only do it once
    point_cloud_preds = svm.predict((point_cloud[features].to_numpy() - train_mean) / train_std)
    svm_acc = (point_cloud_preds == point_cloud.classification.to_numpy()).sum() / len(point_cloud)
    print(f'SVM accuracy: {svm_acc:.2%}')

    if visualize:
        visualize_cloud(
            point_cloud[['x', 'y', 'z']].to_numpy(),
            colors=point_cloud_preds,
            cmap='cool',
            bounds=bounds,
        )

    ###################################################################################################################
    # QSVM w/ Quantum Kernels #########################################################################################
    ###################################################################################################################

    if verbose:
        print('Optimizing QSVM model with Quantum Kernel...')
    qsvm_search_space = {
        'B': [2],
        'P': [0, 1, 2],
        'K': [3, 4, 5],
        'zeta': [0.5, 1.0, 1.5],
        'kernel': kernels,
    }
    qsvm_kw_params = {'sampler': 'steepest_descent', 'num_reads': 100, 'normalize': True}
    params, _ = grid_search(
        QSVM,
        search_space=qsvm_search_space,
        svm_kw_params=qsvm_kw_params,
        train_x=train_x,
        train_y=train_y,
        valid_x=valid_x,
        valid_y=valid_y,
        processes=1,  # `qiskit_machine_learning.kernels.FidelityStatevectorKernel` is not picklable
        # save_path=WORKING_DIR / 'results2' / f'{SEED}_QSVM_results_{IID}.txt',
        save_path=None,
    )
    if verbose:
        print(f'QSVM params: {params | qsvm_kw_params}')
    qsvm = QSVM(**params, **qsvm_kw_params)
    qsvm.fit(train_x, train_y)

    qsvm_acc = qsvm.score(valid_x, valid_y)
    print(f'QSVM validation accuracy: {qsvm_acc:.2%}')

    # Predicting with a quantum kernel is expensive due to the number of circuits that must be simulated,
    # so we only do it once
    point_cloud_preds = qsvm(point_cloud[features].to_numpy())
    qsvm_acc = (point_cloud_preds == point_cloud.classification.to_numpy()).sum() / len(point_cloud)
    print(f'QSVM accuracy: {qsvm_acc:.2%}')

    if visualize:
        visualize_cloud(
            point_cloud[['x', 'y', 'z']].to_numpy(),
            colors=point_cloud_preds,
            cmap='cool',
            bounds=bounds,
        )

    ###################################################################################################################
    # Weak Classifiers ################################################################################################
    ###################################################################################################################

    n_ensemble_train_samples = 1000
    n_ensemble_valid_samples = 1000

    indices = list(range(len(point_cloud)))
    random.shuffle(indices)
    train_indices = indices[:n_ensemble_train_samples]
    valid_indices = indices[-n_ensemble_valid_samples:]
    train_x_ensemble = point_cloud[features].iloc[train_indices].to_numpy()
    valid_x_ensemble = point_cloud[features].iloc[valid_indices].to_numpy()
    train_y_ensemble = point_cloud.classification.iloc[train_indices].to_numpy()
    valid_y_ensemble = point_cloud.classification.iloc[valid_indices].to_numpy()

    n_weak_classifiers = 50
    samples_per_classifier = 20

    # Split the training data into `n_weak_classifiers` random subsets of size `samples_per_classifier`
    split_indices = [
        random.sample(range(n_ensemble_train_samples), k=samples_per_classifier) for _ in range(n_weak_classifiers)
    ]
    train_x_split = [train_x_ensemble[ii] for ii in split_indices]
    train_y_split = [train_y_ensemble[ii] for ii in split_indices]

    if verbose:
        print(f'Optimizing {n_weak_classifiers} classifiers with {samples_per_classifier} samples per classifier...')
    # Perform a grid search to optimize each weak classifier
    weak_clf_search_space = {
        'B': [2],
        'P': [0, 1, 2],
        'K': [2, 3, 4],
        'zeta': [0.5, 1, 1.5],
        'gamma': np.geomspace(0.1, 1, 5),
    }
    weak_clf_kw_params = {'kernel': 'rbf', 'sampler': 'steepest_descent', 'num_reads': 100, 'normalize': True}
    weak_clf_params = [
        grid_search(
            QSVM,
            weak_clf_search_space,
            tx,
            ty,
            valid_x_ensemble,
            valid_y_ensemble,
            weak_clf_kw_params,
            processes=PROCESSES,
        )[0]
        for tx, ty in zip(train_x_split, train_y_split)
    ]
    weak_classifiers = [QSVM(**params, **weak_clf_kw_params) for params in weak_clf_params]

    if verbose:
        print('Fitting weak classifiers...')
    # Fit the weak classifiers
    for qsvm, x, y in zip(weak_classifiers, train_x_split, train_y_split):
        qsvm.fit(x, y)

    ###################################################################################################################
    # QBoost ##########################################################################################################
    ###################################################################################################################

    if verbose:
        print('Optimizing QBoost model...')
    # Optimize the QBoost classifier
    qboost_search_space = {'B': [2], 'P': [0, 1, 2, 3, 4], 'K': [3, 4, 5, 6, 7, 8]}
    qboost_kw_params = {'weak_classifiers': weak_classifiers, 'lbda': (0.0, 2.1, 0.1), 'num_reads': 100}
    qboost_params, _ = grid_search(
        QBoost,
        qboost_search_space,
        train_x_ensemble,
        train_y_ensemble,
        valid_x_ensemble,
        valid_y_ensemble,
        qboost_kw_params,
        processes=PROCESSES,
    )
    if verbose:
        print(f'QBoost params: {qboost_params | qboost_kw_params}')
    # Define the strong classifier
    qboost = QBoost(**qboost_params, **qboost_kw_params)
    qboost.fit(train_x_ensemble, train_y_ensemble, valid_x_ensemble, valid_y_ensemble)

    qboost_acc = qboost.score(valid_x, valid_y)
    print(f'QBoost validation accuracy: {qboost_acc:.2%}')

    qboost_acc = qboost.score(point_cloud[features].to_numpy(), point_cloud.classification.to_numpy())
    print(f'QBoost accuracy: {qboost_acc:.2%}')

    if visualize:
        visualize_cloud(
            point_cloud[['x', 'y', 'z']].to_numpy(),
            colors=qboost(point_cloud[features].to_numpy()),
            cmap='cool',
            bounds=bounds,
        )

    ###################################################################################################################
    # AdaBoost ########################################################################################################
    ###################################################################################################################

    if verbose:
        print('Optimizing AdaBoost model...')
    # Run the AdaBoost algorithm on the ensemble of weak QSVMs
    best_n_est = None
    best_acc = 0
    for n_est in range(5, 100, 2):
        adaboost = AdaBoost(weak_classifiers=weak_classifiers, n_estimators=n_est)
        adaboost.fit(train_x_ensemble, train_y_ensemble)
        acc = adaboost.score(valid_x_ensemble, valid_y_ensemble)

        if acc > best_acc:
            best_acc = acc
            best_n_est = n_est

    if verbose:
        print(f'AdaBoost params: {dict(n_estimators=best_n_est)}')
    # Define the strong classifier
    adaboost = AdaBoost(weak_classifiers=weak_classifiers, n_estimators=best_n_est)
    adaboost.fit(train_x_ensemble, train_y_ensemble)

    adaboost_acc = adaboost.score(valid_x, valid_y)
    print(f'AdaBoost validation accuracy: {adaboost_acc:.2%}')

    adaboost_acc = adaboost.score(point_cloud[features].to_numpy(), point_cloud.classification.to_numpy())
    print(f'AdaBoost accuracy: {adaboost_acc:.2%}')

    if visualize:
        visualize_cloud(
            point_cloud[['x', 'y', 'z']].to_numpy(),
            colors=adaboost(point_cloud[features].to_numpy()),
            cmap='cool',
            bounds=bounds,
        )


if __name__ == '__main__':
    main()
