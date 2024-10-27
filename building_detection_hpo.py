import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional, Any
from itertools import product
from functools import partial
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor

import dill
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import KFold
from tqdm import tqdm
from dotenv import load_dotenv

from qsvm import QSVM
from softmax_qsvm import SoftmaxQSVM
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
from utils import accuracy, visualize_cloud, confusion_matrix, matthews_corrcoef, f1_score


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


def cross_validation_helper(
    index_tuple: tuple[np.ndarray, np.ndarray],
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: Optional[np.ndarray] = None,
    y_valid: Optional[np.ndarray] = None,
) -> np.ndarray[int]:
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

    return confusion_matrix(clf.predict(x_test_fold), y_test_fold)


def cross_validation(
    model: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    k: int,
    x_valid: Optional[np.ndarray] = None,
    y_valid: Optional[np.ndarray] = None,
    seed: Optional[int] = None,
    num_workers: int = 5,
) -> np.ndarray[int]:
    '''
    Apply k-fold cross validation to the model.
    '''

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    if num_workers > 1:
        cross_validation_helper_partial = partial(
            cross_validation_helper, model=model, x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid
        )
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            cms = list(executor.map(cross_validation_helper_partial, kf.split(x_train)))

    else:
        cms = []
        for index_tuple in kf.split(x_train):
            cm = cross_validation_helper(
                index_tuple=index_tuple,
                model=model,
                x_train=x_train,
                y_train=y_train,
                x_valid=x_valid,
                y_valid=y_valid,
            )
            cms.append(cm)

    aggregate_cm = sum(cms)
    return aggregate_cm


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
    predict_full_dataset: bool = True,
    score_valid: bool = False,
    write_data: bool = True,
    validate_qboost_w_train: bool = False,
) -> Any:
    if verbose:
        print(f'Optimizing {model_name} model...')

    # Divide the search space into param dictionaries we can pass to the model
    param_sets = (dict(zip(search_space.keys(), values)) for values in product(*search_space.values()))
    best_clfs = (-float('inf'), [None])  # (mcc, [params, ...])
    # Hyperparameter optimization loop
    for params in tqdm(
        param_sets,
        desc=f'Optimizing {model_name}',
        total=math.prod(len(x) for x in search_space.values()),
        disable=not verbose,
    ):
        # Instantiate the model
        clf = model(**params, **kw_params)

        # If we are only using one fold, do not use cross-validation
        if k_folds == 1:
            # The fit method for QBoost takes validation data as well
            (
                clf.fit(x_train, y_train, x_valid[: len(x_train)], y_valid[: len(x_train)])
                if model == QBoost and not validate_qboost_w_train
                else clf.fit(x_train, y_train)
            )
            preds = clf.predict(x_valid) if score_valid else clf.predict(x_train)
            cm = confusion_matrix(preds, y_valid if score_valid else y_train)
        else:
            # The fit method for QBoost takes validation data as well
            fit_args = dict(x_train=x_train, y_train=y_train)
            if model == QBoost and not validate_qboost_w_train:
                fit_args |= dict(x_valid=x_valid[: len(x_train)], y_valid=y_valid[: len(x_train)])
            # Fit using cross validation
            cm = cross_validation(model=clf, k=k_folds, **fit_args, num_workers=num_cv_workers)

        mcc = matthews_corrcoef(cm)

        # Update the best known model
        if mcc > best_clfs[0]:
            best_clfs = (mcc, [params])
        elif mcc == best_clfs[0]:
            best_clfs[1].append(params)

        # Write results to a file
        if write_data:
            if model in [SoftmaxQSVM, QBoost, AdaBoost]:
                params_to_print = deepcopy(params)
                weak_classifiers = params_to_print.pop('weak_classifiers')
                params_to_print['S'] = len(weak_classifiers)
                params_to_print['M'] = len(weak_classifiers[0].X_)
            else:
                params_to_print = params
            with open(HPO_LOG_DIR / f'{model_name.lower().replace(" ", "_")}_{SEED}.txt', 'a') as f:
                print(f'{mcc:.4f};{";".join(map(str, cm))};{params_to_print | kw_params}', file=f)

    # The best accuracy, and a list of the parameter sets that provide that accuracy
    mcc, param_sets = best_clfs

    if verbose:
        print(f'{model_name} best params with training MCC {mcc:.4f}:')
        for params in param_sets:
            print(f'\t{params | kw_params}')

    # Define and train the best classifier on the whole training set
    best_clf = model(**param_sets[0], **kw_params)
    (
        best_clf.fit(x_train, y_train, x_valid[: len(x_train)], y_valid[: len(x_train)])
        if model == QBoost and not validate_qboost_w_train
        else best_clf.fit(x_train, y_train)
    )

    # Predict and print results
    train_cm = confusion_matrix(best_clf.predict(x_train), y_train)
    print(f'{model_name} training MCC: {matthews_corrcoef(train_cm):.4f}')
    print(f'{model_name} training F1:  {f1_score(train_cm):.4f}')
    print(f'{model_name} training acc: {accuracy(train_cm):.2%}')
    valid_cm = confusion_matrix(best_clf.predict(x_valid), y_valid)
    print(f'{model_name} validation MCC: {matthews_corrcoef(valid_cm):.4f}')
    print(f'{model_name} validation F1:  {f1_score(valid_cm):.4f}')
    print(f'{model_name} validation acc: {accuracy(valid_cm):.2%}')

    if predict_full_dataset:
        point_cloud_preds = best_clf.predict(x_all)
        full_cm = confusion_matrix(point_cloud_preds, y_all)
        print(f'{model_name} full dataset MCC: {matthews_corrcoef(full_cm):.4%}')
        print(f'{model_name} full dataset F1:  {f1_score(full_cm):.4f}')
        print(f'{model_name} full dataset acc: {accuracy(full_cm):.2%}')

        if visualize:
            visualize_cloud(point_cloud[['x', 'y', 'z']].to_numpy(), colors=point_cloud_preds, cmap='cool')

    return clf


def train_weak_classifiers_hpo(
    train_x: np.ndarray,
    train_y: np.ndarray,
    SM: list[tuple[int, int]],
    balance_classes: bool,
    verbose: bool = True,
) -> list[list[QSVM]]:

    # SM = (Number of classifiers, Size of subsets)

    weak_classifiers = []
    for S, M in SM:
        # Split the training data into `n_weak_classifiers` random subsets of size `samples_per_classifier`.
        # Use a SoftmaxQSVM to accomplish this.
        softmax_qsvm = SoftmaxQSVM(None, {}, S=S, M=M, balance_classes=balance_classes, multiplier=1.0)
        softmax_qsvm._define_data_subsets(train_x, train_y)
        train_x_split = softmax_qsvm._x_subsets
        train_y_split = softmax_qsvm._y_subsets

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
            best_weak_qsvm = (None, None, -float('inf'))  # (qsvm, params, mcc)
            # Hyperparameter optimization loop
            for params in param_sets:
                qsvm = QSVM(**params, **weak_clf_kw_params)
                qsvm.fit(tx, ty)
                preds = qsvm.predict(tx)
                cm = confusion_matrix(preds, ty)
                mcc = matthews_corrcoef(cm)
                if mcc > best_weak_qsvm[2]:
                    best_weak_qsvm = (qsvm, params, mcc)

            qsvm, params, _ = best_weak_qsvm
            weak_clfs.append(qsvm)
        weak_classifiers.append(weak_clfs)

    return weak_classifiers


def get_data(
    n_train_samples: int,
    n_valid_samples: int,
    features: list[str],
    verbose: bool,
    visualize: bool,
    dataset: str,
    working_dir: Path,
) -> tuple[
    pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    if dataset == 'kits':
        data_file = working_dir / 'data' / '4870E_54560N_kits' / '1m_lidar.csv'
    elif dataset == 'downtown':
        data_file = working_dir / 'data' / '491000_5458000_downtown' / '1m_lidar.csv'
    elif dataset == 'ptgrey':
        data_file = working_dir / 'data' / '483000_5457000_ptgrey' / '1m_lidar.csv'
    point_cloud = pd.read_csv(data_file)

    if n_train_samples + n_valid_samples > len(point_cloud):
        raise ValueError(
            f'Cannot sample {n_train_samples + n_valid_samples} samples from point cloud of size {len(point_cloud)}'
        )

    # Map building points to 1 and all others to -1
    point_cloud.classification = point_cloud.classification.map(lambda x: 1 if x == 6 else -1)

    if verbose:
        value_counts = point_cloud.classification.value_counts()
        print(
            'Point cloud:'
            f'\n\tTotal # points:        {len(point_cloud):_}'
            f'\n\t# building points:     {value_counts.get(1, None):_}'
            f'\n\t# non-building points: {value_counts.get(-1, None):_}'
        )

    pc = downsample_point_cloud(point_cloud, factor=9 / 16, keep_max=True) if dataset == 'ptgrey' else point_cloud

    train_y = np.empty(0)
    while -1 not in train_y or 1 not in train_y:
        indices = np.arange(len(pc))
        np.random.shuffle(indices)
        train_indices = indices[:n_train_samples]
        valid_indices = indices[-n_valid_samples:]

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
            f'\n\tTotal # points:        {n_train_samples:_}'
            f'\n\t# building points:     {sum(1 for y in train_y if y == 1):_}'
            f'\n\t# non-building points: {sum(1 for y in train_y if y == -1):_}'
        )
        print(
            'Validation set:'
            f'\n\tTotal # points:        {n_valid_samples:_}'
            f'\n\t# building points:     {sum(1 for y in valid_y if y == 1):_}'
            f'\n\t# non-building points: {sum(1 for y in valid_y if y == -1):_}'
        )

    if visualize:
        visualize_cloud(pc[['x', 'y', 'z']].to_numpy(), colors=pc.classification.to_numpy(), cmap='cool')

    return (
        point_cloud,
        train_x,
        train_y,
        valid_x,
        valid_y,
        train_x_normalized,
        valid_x_normalized,
        train_mean,
        train_std,
    )


def hpo():
    if SEED is not None:
        print(f'Using {SEED = }')
        np.random.seed(SEED)

    predict_full_dataset = False
    verbose = True
    visualize = False
    num_softmax_qsvm_workers = 4
    write_data = True
    k_folds = 3
    num_cv_workers = 3

    # Choose a random subset of points as a train set
    n_train_samples = 1_000
    n_valid_samples = 100_000

    features = ['z', 'normal_variation', 'height_variation', 'log_intensity']

    point_cloud, train_x, train_y, valid_x, valid_y, train_x_normalized, valid_x_normalized, train_mean, train_std = (
        get_data(n_train_samples, n_valid_samples, features, verbose, visualize, DATASET, WORKING_DIR)
    )

    ###################################################################################################################
    # SVM #############################################################################################################
    ###################################################################################################################

    if MODELS is None or '0' in MODELS:
        svm_search_space = {'C': np.geomspace(0.01, 10, 13), 'gamma': np.geomspace(0.01, 10, 13)}
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
            k_folds=k_folds,
            num_cv_workers=1,
            point_cloud=point_cloud,
            model_name='SVM',
            verbose=verbose,
            visualize=visualize,
            predict_full_dataset=predict_full_dataset,
            score_valid=False,
            write_data=write_data,
        )

    ###################################################################################################################
    # QSVM ############################################################################################################
    ###################################################################################################################

    if MODELS is None or '1' in MODELS:
        qsvm_search_space = {
            'B': [2],
            'P': [0, 1, 2],
            'K': [3, 4, 5],
            'zeta': [0.0, 0.4, 0.8, 1.2],
            'gamma': np.geomspace(0.01, 10, 13),
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
            k_folds=k_folds,
            num_cv_workers=num_cv_workers,
            point_cloud=point_cloud,
            model_name='QSVM',
            verbose=verbose,
            visualize=visualize,
            predict_full_dataset=predict_full_dataset,
            score_valid=False,
            write_data=write_data,
        )

    ###################################################################################################################
    # SVM w/ Quantum Kernel ###########################################################################################
    ###################################################################################################################

    if MODELS is None or '2' in MODELS or '3' in MODELS:
        n_features = 4

        if verbose:
            print('Defining quantum kernels...')
        uninitialized_kernels = []
        for reps in [1]:
            for entanglement in ['linear', 'pairwise']:
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
            for entanglement in ['full', 'linear']:
                uninitialized_kernels += [
                    (iqp_feature_map, dict(num_features=n_features, reps=reps, entanglement=entanglement)),
                    (
                        data_reuploading_feature_map,
                        dict(num_features=n_features, reps=reps, entanglement=entanglement),
                    ),
                ]
            uninitialized_kernels += [
                (polynomial_feature_map, dict(num_features=n_features, qubits_per_feature=reps)),
                (qaoa_inspired_feature_map, dict(num_features=n_features, reps=reps)),
            ]

        kernels = []
        for kernel in uninitialized_kernels:
            kernels.append(
                Kernel(
                    kernel[0](**kernel[1]),
                    f'{kernel[0].__name__}({", ".join(k + "=" + str(v) for k, v in kernel[1].items())})',
                )
            )

    if MODELS is None or '2' in MODELS:
        kernel_svm_search_space = {'C': np.geomspace(0.01, 10, 13), 'kernel': kernels}
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
            k_folds=k_folds,
            num_cv_workers=num_cv_workers,
            point_cloud=point_cloud,
            model_name='Quantum Kernel SVM',
            verbose=verbose,
            visualize=visualize,
            predict_full_dataset=predict_full_dataset,
            score_valid=False,
            write_data=write_data,
        )

    ###################################################################################################################
    # QSVM w/ Quantum Kernels #########################################################################################
    ###################################################################################################################

    if MODELS is None or '3' in MODELS:
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
            k_folds=k_folds,
            num_cv_workers=num_cv_workers,
            point_cloud=point_cloud,
            model_name='Quantum Kernel QSVM',
            verbose=verbose,
            visualize=visualize,
            predict_full_dataset=predict_full_dataset,
            score_valid=False,
            write_data=write_data,
        )

    ###################################################################################################################
    # Weak Classifiers ################################################################################################
    ###################################################################################################################

    # Ensemble hyperparameters
    SM = [(20, 50), (50, 20), (40, 40)]  # (Number of classifiers, Size of subsets)
    balance_classes = True

    if MODELS is None or '4' in MODELS or '5' in MODELS or '6' in MODELS:
        weak_classifiers = train_weak_classifiers_hpo(train_x, train_y, SM, balance_classes, verbose)

    ###################################################################################################################
    # Softmax QSVM ####################################################################################################
    ###################################################################################################################

    if MODELS is None or '4' in MODELS:
        softmax_qsvm_kw_params = {'num_workers': num_softmax_qsvm_workers}
        softmax_qsvm_search_space = {
            'multiplier': np.geomspace(0.1, 10, 5),
            'weak_classifiers': weak_classifiers,
        }

        optimize_model(
            model=SoftmaxQSVM,
            search_space=softmax_qsvm_search_space,
            kw_params=softmax_qsvm_kw_params,
            x_train=train_x,
            y_train=train_y,
            x_valid=valid_x,
            y_valid=valid_y,
            x_all=point_cloud[features].to_numpy(),
            y_all=point_cloud.classification.to_numpy(),
            k_folds=k_folds,
            num_cv_workers=num_cv_workers,
            point_cloud=point_cloud,
            model_name='Softmax QSVM',
            verbose=verbose,
            visualize=visualize,
            predict_full_dataset=predict_full_dataset,
            score_valid=False,
            write_data=write_data,
        )

    ###################################################################################################################
    # QBoost ##########################################################################################################
    ###################################################################################################################

    if MODELS is None or '5' in MODELS:
        qboost_search_space = {
            'B': [2],
            'P': [0, 1, 2, 3],
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
            predict_full_dataset=predict_full_dataset,
            score_valid=False,
            write_data=write_data,
            validate_qboost_w_train=True,
        )

    ###################################################################################################################
    # AdaBoost ########################################################################################################
    ###################################################################################################################

    if MODELS is None or '6' in MODELS:
        adaboost_search_space = {'n_estimators': list(range(6, 81, 2)), 'weak_classifiers': weak_classifiers}
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
            predict_full_dataset=predict_full_dataset,
            score_valid=False,
            write_data=write_data,
        )


def compute_qsvm_threshold():
    if SEED is not None:
        print(f'Using {SEED = }')
        np.random.seed(SEED)

    # Choose a random subset of points as a train set
    n_train_samples = 1_000
    n_valid_samples = 100_000

    features = ['z', 'normal_variation', 'height_variation', 'log_intensity']

    # Best params for the QSVM model
    if DATASET == 'kits':
        params = dict(B=2, P=1, K=4, zeta=0.0, gamma=0.1778279410038923)
    elif DATASET == 'downtown':
        params = dict(B=2, P=2, K=4, zeta=0.4, gamma=1.0)
    elif DATASET == 'ptgrey':
        params = dict(B=2, P=2, K=4, zeta=1.2, gamma=1.7782794100389228)
    params |= dict(kernel='rbf', sampler='steepest_descent', num_reads=100, normalize=True, optimize_memory=False)

    n_trials = 200
    threshold_space = np.hstack(([0.0], np.geomspace(0.005, 0.75, 30)))
    ratio = threshold_space[2] / threshold_space[1]
    threshold_space = np.geomspace(0.005 / ratio**23, 0.75 / ratio**30, 23)

    cm_dtype = np.float64

    all_n_accepted = np.empty((n_trials, len(threshold_space)), dtype=int)
    all_n_rejected = np.empty((n_trials, len(threshold_space)), dtype=int)
    all_cms = np.empty((n_trials, len(threshold_space), 4), dtype=cm_dtype)

    for trial in tqdm(range(n_trials)):
        _, train_x, train_y, valid_x, valid_y, _, _, _, _ = get_data(
            n_train_samples, n_valid_samples, features, False, False, DATASET, WORKING_DIR
        )

        n_accepted, n_rejected = np.empty(len(threshold_space), dtype=int), np.empty(len(threshold_space), dtype=int)
        cms = np.empty((len(threshold_space), 4), dtype=cm_dtype)

        for i, threshold in enumerate(threshold_space):
            params['threshold'] = threshold
            params['threshold_strategy'] = 'relative'

            qsvm = QSVM(**params)
            qsvm.fit(train_x, train_y)

            n_rejected[i] = qsvm.n_rejected_
            n_accepted[i] = qsvm.n_accepted_

            cms[i] = confusion_matrix(qsvm.predict(valid_x), valid_y).astype(cm_dtype)

        all_n_accepted[trial] = n_accepted
        all_n_rejected[trial] = n_rejected
        all_cms[trial] = cms

    with open(
        LOG_DIR / f'threshold_analysis_ts={n_train_samples}' / f'threshold_analysis_{DATASET}_{SEED}.pkl', 'wb'
    ) as f:
        dill.dump((threshold_space, all_n_accepted, all_n_rejected, all_cms), f)


def analyze_qsvm_threshold():
    import matplotlib.pyplot as plt
    from matplotlib.ticker import SymmetricalLogLocator, LogLocator

    plt.rcParams.update(
        {
            'text.usetex': True,
            'font.family': 'serif',
            'font.serif': ['Computer Modern Roman'],
        }
    )

    thresh_space1 = np.hstack(([0.0], np.geomspace(0.005, 0.75, 30)))
    ratio = thresh_space1[2] / thresh_space1[1]
    thresh_space2 = np.geomspace(0.005 / ratio**23, 0.75 / ratio**30, 23)
    full_thresh_space = np.hstack(([0.0], np.geomspace(0.005 / ratio**23, 0.75, 53)))

    _, axs = plt.subplots(2, 1, figsize=(8, 8 * 0.8))
    for i, dataset in enumerate(('kits', 'downtown', 'ptgrey')):
        cms = np.full((200, 54, 4), -1, dtype=np.float64)
        for file in (LOG_DIR / 'threshold_analysis_ts=1000').glob(f'threshold_analysis_{dataset}_[0-9]*.pkl'):
            with open(file, 'rb') as f:
                thresh_space, n_accepted, n_rejected, cm = dill.load(f)
            if np.array_equal(thresh_space, thresh_space1):
                slice1 = (slice(100) if cms[0, 0, 0] < 0 else slice(100, None), 0, slice(None))
                slice2 = (slice(100) if cms[0, 0, 0] < 0 else slice(100, None), slice(24, None), slice(None))
                assert np.all(np.isin(cms[slice1], (-1,))), 'Slice 1 indexing incorrect'
                assert np.all(np.isin(cms[slice2], (-1,))), 'Slice 2 indexing incorrect'
                cms[slice1] = cm[:, 0, :]
                cms[slice2] = cm[:, 1:, :]
            elif np.array_equal(thresh_space, thresh_space2):
                slice3 = np.s_[:, 1:24, :]
                assert np.all(np.isin(cms[slice3], (-1,))), 'Slice 3 indexing incorrect'
                cms[slice3] = cm
            else:
                raise ValueError(f'Unknown threshold space: {thresh_space}')
        assert np.all(cms >= 0), 'CM array not set correctly'

        # mccs = np.fromiter((matthews_corrcoef(cm) for cm in cms), dtype=float)
        # accs = np.fromiter((accuracy(cm) for cm in cms), dtype=float)
        mccs = np.apply_along_axis(matthews_corrcoef, 2, cms)
        accs = np.apply_along_axis(accuracy, 2, cms)
        mcc_avg = mccs.mean(axis=0)
        acc_avg = accs.mean(axis=0)
        mcc_std = mccs.std(axis=0)
        acc_std = accs.std(axis=0)

        axs[0].errorbar(
            full_thresh_space,
            mcc_avg,
            yerr=mcc_std,
            fmt='.',
            c=['firebrick', 'darkblue', 'darkgreen'][i],
            linewidth=1,
            label=f'Area {i + 1}',
            capsize=2,
        )
        axs[1].errorbar(
            full_thresh_space,
            acc_avg,
            yerr=acc_std,
            fmt='.',
            c=['firebrick', 'darkblue', 'darkgreen'][i],
            linewidth=1,
            label=f'Area {i + 1}',
            capsize=2,
        )

    for ax in axs:
        ax.set_xlabel(r'Relative Threshold $T$')
    axs[0].set_ylabel(r'MCC')
    axs[1].set_ylabel(r'Accuracy')

    # Set x-axis to symlog scale
    # axs[0].set_xscale('symlog', linthresh=full_thresh_space[1], linscale=0.2)
    for ax in axs:
        ax.set_xscale('symlog', linthresh=1e-4, linscale=0.2)
        ax.set_xlim(left=-0.00002, right=1)  # right=axs[0].get_xlim()[1])

        # # Customize x-axis ticks
        ax.xaxis.set_major_locator(SymmetricalLogLocator(linthresh=full_thresh_space[1], base=10))
        ax.xaxis.set_minor_locator(LogLocator(subs='all'))

        major_xticks = ax.xaxis.get_major_ticks()
        major_xticks[1].set_visible(False)

        minor_xticks = ax.xaxis.get_minor_ticks()
        for tick in minor_xticks[:8]:
            tick.set_visible(False)

        ax.grid(which='both', alpha=0.3)
        ax.legend(loc='lower left')

    axs[0].set_ylim([0.36, 0.74])
    plt.tight_layout()
    plt.savefig(WORKING_DIR / 'plots' / f'threshold_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()


if __name__ == '__main__':
    # Set DWave API token
    load_dotenv()

    DATASET = sys.argv[1] if len(sys.argv) > 1 else None
    assert DATASET is None or DATASET in ('kits', 'downtown', 'ptgrey')

    WORKING_DIR = Path(__file__).parent
    LOG_DIR = WORKING_DIR / 'logs'
    if DATASET is not None:
        HPO_LOG_DIR = LOG_DIR / f'hpo_logs_{DATASET}'

    MODELS = sys.argv[2] if len(sys.argv) > 2 else None

    SEED = int(sys.argv[3]) if len(sys.argv) > 3 else np.random.randint(100, 1_000_000)

    hpo()
    # compute_qsvm_threshold()
    # analyze_qsvm_threshold()
