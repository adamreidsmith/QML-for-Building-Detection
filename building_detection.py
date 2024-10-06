import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional, Any
from itertools import product
from functools import partial
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
from tqdm import tqdm
from dotenv import load_dotenv

# from qsvm import QSVM, QSVMGroup
from qsvm import QSVM
from qsvm_group import QSVMGroup
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

# DATASET = 'kits'
# DATASET = 'downtown'
# DATASET = 'ptgrey'
DATASET = sys.argv[1]
assert DATASET in ('kits', 'downtown', 'ptgrey')

WORKING_DIR = Path(__file__).parent
LOG_DIR = WORKING_DIR / f'logs_{DATASET}'

# SEED = int(sys.argv[1]) if len(sys.argv) > 1 else np.random.randint(100, 1_000_000)
SEED = np.random.randint(100, 1_000_000)


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


def downsample_point_cloud(point_cloud: pd.DataFrame, factor: float, keep_max: bool) -> pd.DataFrame:
    '''
    Crop the point cloud to factor * its original area.
    '''

    xmin, xmax, ymin, ymax = point_cloud.x.min(), point_cloud.x.max(), point_cloud.y.min(), point_cloud.y.max()
    if keep_max:
        xmin = xmax - (xmax - xmin) * np.sqrt(factor)
        ymin = ymax - (ymax - ymin) * np.sqrt(factor)
    else:
        xmax = xmin + (xmax - xmin) * np.sqrt(factor)
        ymax = ymin + (ymax - ymin) * np.sqrt(factor)

    point_cloud = point_cloud[
        (point_cloud.x >= xmin) & (point_cloud.x <= xmax) & (point_cloud.y >= ymin) & (point_cloud.y <= ymax)
    ]
    point_cloud = point_cloud.reset_index(drop=True)
    return point_cloud


def acc_f1(predictions: np.ndarray, labels: np.ndarray) -> float:
    acc = np.sum(predictions == labels) / labels.shape[0]
    f1 = f1_score(y_true=labels, y_pred=predictions, pos_label=1, average='binary', zero_division=0.0)
    return acc, f1


def cross_validation_helper(
    index_tuple: tuple[np.ndarray, np.ndarray],
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: Optional[np.ndarray] = None,
    y_valid: Optional[np.ndarray] = None,
):
    '''
    Helper function to allow for parallelization of cross validation.
    '''

    clf = deepcopy(model)

    train_indices, test_indices = index_tuple
    x_train_fold, y_train_fold = x_train[train_indices], y_train[train_indices]
    x_test_fold, y_test_fold = x_train[test_indices], y_train[test_indices]

    if x_valid is not None and y_valid is not None:
        clf.fit(x_train_fold, y_train_fold, x_valid, y_valid)
    else:
        clf.fit(x_train_fold, y_train_fold)

    return acc_f1(clf.predict(x_test_fold), y_test_fold)


def cross_validation(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    x_valid: Optional[np.ndarray] = None,
    y_valid: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    num_workers: int = 5,
) -> float:
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    if num_workers > 1:
        cross_validation_helper_partial = partial(
            cross_validation_helper, model=model, x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid
        )
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            accs_and_f1s = list(executor.map(cross_validation_helper_partial, kf.split(x_train)))
        accs, f1s = list(zip(*accs_and_f1s))

    else:
        accs, f1s = [], []
        for index_tuple in kf.split(x_train):
            acc, f1 = cross_validation_helper(
                index_tuple=index_tuple,
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
            )
            accs.append(acc)
            f1s.append(f1)

    avg_acc = np.mean(accs)
    avg_f1 = np.mean(f1s)

    return avg_acc, avg_f1


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
    k_folds: int,
    num_cv_workers: int,
    point_cloud: pd.DataFrame,
    model_name: str = '',
    verbose: bool = True,
    visualize: bool = True,
    model_kw_params: Optional[dict[str, Any]] = None,
    predict_full_dataset: bool = True,
    score_valid: bool = False,
    write_data: bool = True,
) -> Any:
    if verbose:
        print(f'Optimizing {model_name} model...')

    # Divide the search space into param dictionaries we can pass to the model
    param_sets = (dict(zip(search_space.keys(), values)) for values in product(*search_space.values()))
    best_clfs = (-1.0, [None])  # (accuracy, [params, ...])
    # Hyperparameter optimization loop
    for params in tqdm(
        param_sets,
        desc=f'Optimizing {model_name}',
        total=math.prod(len(x) for x in search_space.values()),
        disable=not verbose,
    ):
        # QSVM Group model is initialized diffrently to the others
        if model == QSVMGroup:
            multiplier, SM = params.pop('multiplier'), params.pop('SM')
            clf = model(params | kw_params, **model_kw_params, multiplier=multiplier, S=SM[0], M=SM[1])
            params['multiplier'], params['SM'] = multiplier, SM
        else:
            clf = model(**params, **kw_params)

        # If we are only using one fold, do not use cross-validation
        if k_folds == 1:
            # The fit method for QBoost takes validation data as well
            (
                clf.fit(x_train, y_train, x_valid[: len(x_train)], y_valid[: len(x_train)])
                if model == QBoost
                else clf.fit(x_train, y_train)
            )
            preds = clf.predict(x_valid) if score_valid else clf.predict(x_train)
            acc, f1 = acc_f1(preds, y_valid if score_valid else y_train)
        else:
            # The fit method for QBoost takes validation data as well
            fit_args = dict(x_train=x_train, y_train=y_train)
            if model == QBoost:
                fit_args |= dict(x_valid=x_valid[: len(x_train)], y_valid=y_valid[: len(x_train)])
            # Fit using cross validation
            acc, f1 = cross_validation(model=clf, k=k_folds, **fit_args, num_workers=num_cv_workers)

        # Update the best known model
        if acc > best_clfs[0]:
            best_clfs = (acc, [params])
        elif acc == best_clfs[0]:
            best_clfs[1].append(params)

        # Write results to a file
        if write_data:
            if model in [QBoost, AdaBoost]:
                params_to_print = deepcopy(params)
                weak_classifiers = params_to_print.pop('weak_classifiers')
                params_to_print['S'] = len(weak_classifiers)
                params_to_print['M'] = len(weak_classifiers[0].X_)
            else:
                params_to_print = params
            with open(LOG_DIR / f'{model_name.lower().replace(" ", "_")}_{SEED}.txt', 'a') as f:
                print(f'{acc:.6f}:{f1:.4f}:{params_to_print | kw_params}', file=f)

    # The best accuracy, and a list of the parameter sets that provide that accuracy
    acc, param_sets = best_clfs

    if verbose:
        print(f'{model_name} best params with training accuracy {acc:.2%}:')
        for params in param_sets:
            print(f'\t{params | kw_params}')

    # Define and train the best classifier on the whole training set
    if model == QSVMGroup:
        multiplier, SM = param_sets[0].pop('multiplier'), param_sets[0].pop('SM')
        best_clf = model(param_sets[0] | kw_params, **model_kw_params, multiplier=multiplier, S=SM[0], M=SM[1])
        param_sets[0]['multiplier'], param_sets[0]['SM'] = multiplier, SM
    else:
        best_clf = model(**param_sets[0], **kw_params)
    (
        best_clf.fit(x_train, y_train, x_valid[: len(x_train)], y_valid[: len(x_train)])
        if model == QBoost
        else best_clf.fit(x_train, y_train)
    )

    # Predict and print results
    train_acc, train_f1 = acc_f1(best_clf.predict(x_train), y_train)
    print(f'{model_name} training accuracy: {train_acc:.2%}')
    print(f'{model_name} training F1:       {train_f1:.3f}')
    valid_acc, valid_f1 = acc_f1(best_clf.predict(x_valid), y_valid)
    print(f'{model_name} validation accuracy: {valid_acc:.2%}')
    print(f'{model_name} validation F1:       {valid_f1:.3f}')

    if predict_full_dataset:
        point_cloud_preds = best_clf.predict(x_all)
        full_acc, full_f1 = acc_f1(point_cloud_preds, y_all)
        print(f'{model_name} full dataset accuracy: {full_acc:.2%}')
        print(f'{model_name} full dataset F1:       {full_f1:.3f}')

        if visualize:
            visualize_cloud(point_cloud[['x', 'y', 'z']].to_numpy(), colors=point_cloud_preds, cmap='cool')

    return clf


def train_weak_classifiers(
    train_x: np.ndarray,
    train_y: np.ndarray,
    SM: list[tuple[int, int]],
    balance_classes: bool,
    verbose: bool = True,
) -> list[QSVM]:

    # SM = (Number of classifiers, Size of subsets)
    # n_weak_classifiers = 50
    # samples_per_classifier = 20
    # balance_classes = True

    weak_classifiers = []
    for S, M in SM:
        # Split the training data into `n_weak_classifiers` random subsets of size `samples_per_classifier`.
        # Use a QSVMGroup to accomplish this.
        qsvm_group = QSVMGroup({}, S=S, M=M, balance_classes=balance_classes, multiplier=1.0)
        qsvm_group._define_data_subsets(train_x, train_y)
        train_x_split = qsvm_group._x_subsets
        train_y_split = qsvm_group._y_subsets

        if verbose:
            print(f'Optimizing {S} classifiers with {M} samples per classifier...')
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
        weak_clfs = []
        for tx, ty in tqdm(zip(train_x_split, train_y_split), desc='Optimizing Weak QSVMs', total=S):
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
            weak_clfs.append(qsvm)
        weak_classifiers.append(weak_clfs)

    return weak_classifiers


def main():
    if SEED is not None:
        print(f'Using {SEED = }')
        np.random.seed(SEED)

    if DATASET == 'kits':
        data_file = WORKING_DIR / 'data' / '4870E_54560N_kits' / '1m_lidar.csv'
    elif DATASET == 'downtown':
        data_file = WORKING_DIR / 'data' / '491000_5458000_downtown' / '1m_lidar.csv'
    elif DATASET == 'ptgrey':
        data_file = WORKING_DIR / 'data' / '483000_5457000_ptgrey' / '1m_lidar.csv'
    full_point_cloud = pd.read_csv(data_file)

    predict_full_dataset = False
    verbose = True
    visualize = False
    num_qsvm_group_workers = 4
    write_data = True
    k_folds = 3
    num_cv_workers = 3

    # Map building points to 1 and all others to 0
    full_point_cloud.classification = full_point_cloud.classification.map(lambda x: 1 if x == 6 else -1)
    # visualize_cloud(
    #     # np.hstack((full_point_cloud[['x', 'y']].to_numpy(), np.zeros((len(full_point_cloud), 1)))),
    #     full_point_cloud[['x', 'y', 'z']].to_numpy(),
    #     colors=full_point_cloud.z.to_numpy(),
    #     cmap='viridis',
    # )

    # downsample_factor = 0.25 if 'kits' in str(data_file) else 1.0
    # if downsample_factor < 1.0:
    #     if verbose:
    #         print(f'Downsampling point cloud by factor {downsample_factor}...')
    #     point_cloud = downsample_point_cloud(
    #         full_point_cloud, downsample_factor, keep_max=('ptgrey' in str(data_file))
    #     )
    # else:
    #     point_cloud = full_point_cloud
    point_cloud = full_point_cloud
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
    n_valid_samples = 100_000

    if DATASET == 'ptgrey':
        pc = downsample_point_cloud(point_cloud, factor=0.25, keep_max=True)
    else:
        pc = point_cloud
    train_y = np.empty(0)
    # Ensure we have sufficiently many building points in the train set
    while len(train_y[train_y == 1]) < n_train_samples / 20:
        indices = np.arange(len(pc))
        np.random.shuffle(indices)
        train_indices = indices[:n_train_samples]
        valid_indices = indices[-n_valid_samples:]

        features = ['z', 'normal_variation', 'height_variation', 'log_intensity']

        train_x = pc[features].iloc[train_indices].to_numpy()
        valid_x = pc[features].iloc[valid_indices].to_numpy()
        train_y = pc.classification.iloc[train_indices].to_numpy()
        valid_y = pc.classification.iloc[valid_indices].to_numpy()

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
            point_cloud[['x', 'y', 'z']].to_numpy(), colors=point_cloud.classification.to_numpy(), cmap='cool'
        )

    ###################################################################################################################
    # SVM #############################################################################################################
    ###################################################################################################################

    # svm_search_space = {'C': np.geomspace(0.001, 100, 16), 'gamma': np.geomspace(0.001, 100, 16)}
    # svm_kw_params = {'class_weight': 'balanced', 'kernel': 'rbf'}

    # optimize_model(
    #     model=SVC,
    #     search_space=svm_search_space,
    #     kw_params=svm_kw_params,
    #     x_train=train_x_normalized,
    #     y_train=train_y,
    #     x_valid=valid_x_normalized,
    #     y_valid=valid_y,
    #     x_all=(point_cloud[features].to_numpy() - train_mean) / train_std,
    #     y_all=point_cloud.classification.to_numpy(),
    #     k_folds=k_folds,
    #     num_cv_workers=1,
    #     point_cloud=point_cloud,
    #     model_name='SVM',
    #     verbose=verbose,
    #     visualize=visualize,
    #     model_kw_params=None,
    #     predict_full_dataset=predict_full_dataset,
    #     score_valid=False,
    #     write_data=write_data,
    # )

    # ###################################################################################################################
    # # QSVM ############################################################################################################
    # ###################################################################################################################

    # qsvm_search_space = {
    #     'B': [2],
    #     'P': [0, 1, 2],
    #     'K': [3, 4, 5],
    #     'zeta': [0.0, 0.4, 0.8, 1.2],
    #     'gamma': np.geomspace(0.1, 1, 5),
    # }
    # qsvm_kw_params = {'kernel': 'rbf', 'sampler': 'steepest_descent', 'num_reads': 100, 'normalize': True}

    # optimize_model(
    #     model=QSVM,
    #     search_space=qsvm_search_space,
    #     kw_params=qsvm_kw_params,
    #     x_train=train_x,
    #     y_train=train_y,
    #     x_valid=valid_x,
    #     y_valid=valid_y,
    #     x_all=point_cloud[features].to_numpy(),
    #     y_all=point_cloud.classification.to_numpy(),
    #     k_folds=k_folds,
    #     num_cv_workers=num_cv_workers,
    #     point_cloud=point_cloud,
    #     model_name='QSVM',
    #     verbose=verbose,
    #     visualize=visualize,
    #     model_kw_params=None,
    #     predict_full_dataset=predict_full_dataset,
    #     score_valid=False,
    #     write_data=write_data,
    # )

    # ###################################################################################################################
    # # QSVM Group ######################################################################################################
    # ###################################################################################################################

    # QSVM Group hyperparameters
    SM = [(20, 50), (50, 20), (40, 40)]  # (Number of classifiers, Size of subsets)
    balance_classes = True

    model_kw_params = {'balance_classes': balance_classes, 'num_workers': num_qsvm_group_workers}
    qsvm_group_search_space = {
        'B': [2],
        'P': [0, 1],
        'K': [4, 5, 6],
        'zeta': [0.0, 0.4, 0.8, 1.2],
        'gamma': np.geomspace(0.01 * np.sqrt(10), 10, 6),
        'multiplier': np.geomspace(0.1, 10, 5),
        'SM': SM,
    }
    qsvm_group_kw_params = {
        'kernel': 'rbf',
        'sampler': 'steepest_descent',
        'num_reads': 100,
        'normalize': True,
    }

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
        k_folds=k_folds,
        num_cv_workers=num_cv_workers,
        point_cloud=point_cloud,
        model_name='QSVM Group',
        verbose=verbose,
        visualize=visualize,
        model_kw_params=model_kw_params,
        predict_full_dataset=predict_full_dataset,
        score_valid=False,
        write_data=write_data,
    )

    ###################################################################################################################
    # SVM w/ Quantum Kernel ###########################################################################################
    ###################################################################################################################

    # n_features = 4

    # if verbose:
    #     print('Defining quantum kernels...')
    # uninitialized_kernels = []
    # for reps in [1]:
    #     for entanglement in ['linear', 'pairwise']:
    #         for single_pauli in 'XYZ':
    #             for double_pauli in ['XX', 'YY', 'ZZ']:
    #                 uninitialized_kernels.append(
    #                     (
    #                         pauli_feature_map,
    #                         dict(
    #                             num_features=n_features,
    #                             paulis=[single_pauli, double_pauli],
    #                             reps=reps,
    #                             entanglement=entanglement,
    #                         ),
    #                     )
    #                 )
    #     for entanglement in ['full', 'linear']:
    #         uninitialized_kernels += [
    #             (iqp_feature_map, dict(num_features=n_features, reps=reps, entanglement=entanglement)),
    #             (data_reuploading_feature_map, dict(num_features=n_features, reps=reps, entanglement=entanglement)),
    #         ]
    #     uninitialized_kernels += [
    #         (polynomial_feature_map, dict(num_features=n_features, qubits_per_feature=reps)),
    #         (qaoa_inspired_feature_map, dict(num_features=n_features, reps=reps)),
    #     ]

    # kernels = []
    # for kernel in uninitialized_kernels:
    #     # kernels.append(Kernel(kernel[0](**kernel[1]), f'{kernel[0].__name__}({kernel[1]})'))
    #     kernels.append(
    #         Kernel(
    #             kernel[0](**kernel[1]),
    #             f'{kernel[0].__name__}({", ".join(k + "=" + str(v) for k, v in kernel[1].items())})',
    #         )
    #     )

    # kernel_svm_search_space = {'C': np.geomspace(0.01, 100, 13), 'kernel': kernels}
    # kernel_svm_kw_params = {'class_weight': 'balanced'}

    # optimize_model(
    #     model=SVC,
    #     search_space=kernel_svm_search_space,
    #     kw_params=kernel_svm_kw_params,
    #     x_train=train_x_normalized,
    #     y_train=train_y,
    #     x_valid=valid_x_normalized,
    #     y_valid=valid_y,
    #     x_all=(point_cloud[features].to_numpy() - train_mean) / train_std,
    #     y_all=point_cloud.classification.to_numpy(),
    #     k_folds=k_folds,
    #     num_cv_workers=num_cv_workers,
    #     point_cloud=point_cloud,
    #     model_name='Quantum Kernel SVM',
    #     verbose=verbose,
    #     visualize=visualize,
    #     model_kw_params=None,
    #     predict_full_dataset=predict_full_dataset,
    #     score_valid=False,
    #     write_data=write_data,
    # )

    # ###################################################################################################################
    # # QSVM w/ Quantum Kernels #########################################################################################
    # ###################################################################################################################

    # if verbose:
    #     print('Optimizing QSVM model with Quantum Kernel...')
    # kernel_qsvm_search_space = {
    #     'B': [2],
    #     'P': [0, 1],
    #     'K': [3, 4, 5],
    #     'zeta': [0.0, 0.4, 0.8, 1.2],
    #     'kernel': kernels,
    # }
    # kernel_qsvm_kw_params = {'sampler': 'steepest_descent', 'num_reads': 100, 'normalize': True}

    # optimize_model(
    #     model=QSVM,
    #     search_space=kernel_qsvm_search_space,
    #     kw_params=kernel_qsvm_kw_params,
    #     x_train=train_x,
    #     y_train=train_y,
    #     x_valid=valid_x,
    #     y_valid=valid_y,
    #     x_all=point_cloud[features].to_numpy(),
    #     y_all=point_cloud.classification.to_numpy(),
    #     k_folds=k_folds,
    #     num_cv_workers=num_cv_workers,
    #     point_cloud=point_cloud,
    #     model_name='Quantum Kernel QSVM',
    #     verbose=verbose,
    #     visualize=visualize,
    #     model_kw_params=None,
    #     predict_full_dataset=predict_full_dataset,
    #     score_valid=False,
    #     write_data=write_data,
    # )

    # ###################################################################################################################
    # # QSVM Group w/ Quantum Kernels ###################################################################################
    # ###################################################################################################################

    # model_kw_params = {'balance_classes': balance_classes, 'num_workers': num_qsvm_group_workers}
    # kernel_qsvm_group_search_space = {
    #     'B': [2],
    #     'P': [0, 1],
    #     'K': [3, 4, 5],
    #     'zeta': [0.0, 0.4, 0.8, 1.2],
    #     'kernel': kernels,
    #     'multiplier': [1.0],
    #     'SM': SM,
    # }
    # kernel_qsvm_group_kw_params = {
    #     'sampler': 'steepest_descent',
    #     'num_reads': 100,
    #     'normalize': True,
    #     'balance_classes': balance_classes,
    # }

    # optimize_model(
    #     model=QSVMGroup,
    #     search_space=kernel_qsvm_group_search_space,
    #     kw_params=kernel_qsvm_group_kw_params,
    #     x_train=train_x,
    #     y_train=train_y,
    #     x_valid=valid_x,
    #     y_valid=valid_y,
    #     x_all=point_cloud[features].to_numpy(),
    #     y_all=point_cloud.classification.to_numpy(),
    #     k_folds=k_folds,
    #     num_cv_workers=num_cv_workers,
    #     point_cloud=point_cloud,
    #     model_name='Quantum Kernel QSVM Group',
    #     verbose=verbose,
    #     visualize=visualize,
    #     model_kw_params=model_kw_params,
    #     predict_full_dataset=predict_full_dataset,
    #     score_valid=False,
    #     write_data=write_data,
    # )

    ###################################################################################################################
    # Weak Classifiers ################################################################################################
    ###################################################################################################################

    weak_classifiers = train_weak_classifiers(train_x, train_y, SM, balance_classes, verbose)

    ###################################################################################################################
    # QBoost ##########################################################################################################
    ###################################################################################################################

    qboost_search_space = {
        'B': [2],
        'P': [0, 1, 2, 3, 4],
        'K': [3, 4, 5, 6, 7, 8],
        'weak_classifiers': weak_classifiers,
    }
    qboost_kw_params = {'lbda': (0.0, 2.1, 0.1), 'num_reads': 100}

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
        k_folds=k_folds,
        num_cv_workers=num_cv_workers,
        point_cloud=point_cloud,
        model_name='QBoost',
        verbose=verbose,
        visualize=visualize,
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
    adaboost_search_space = {'n_estimators': list(range(6, 80, 2)), 'weak_classifiers': weak_classifiers}
    adaboost_kw_params = {}

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
        k_folds=k_folds,
        num_cv_workers=1,
        point_cloud=point_cloud,
        model_name='AdaBoost',
        verbose=verbose,
        visualize=visualize,
        model_kw_params=None,
        predict_full_dataset=predict_full_dataset,
        score_valid=False,
        write_data=write_data,
    )


if __name__ == '__main__':
    main()
