import sys
from copy import deepcopy
from pathlib import Path
from typing import Optional

import dill
import numpy as np
from tqdm import tqdm

from qsvm import QSVM
from softmax_qsvm import SoftmaxQSVM
from qboost import QBoost
from adaboost import AdaBoost
from utils import accuracy, confusion_matrix, matthews_corrcoef, f1_score
from building_detection_hpo import get_data


def train_weak_qsvms(
    qsvm_params: dict,
    x_train: np.ndarray,
    y_train: np.ndarray,
    S: int,
    M: int,
    balance_classes: bool,
    save: Optional[Path] = None,
    load: Optional[Path] = None,
) -> list[QSVM]:
    '''
    S : int
        Number of classifiers.
    M : int
        Number of samples per subset.
    '''

    if load is not None:
        with open(load, 'rb') as f:
            data = dill.load(f)  # data = (x_train_subsets, y_train_subsets, qsvm_params, weak_classifiers)
        return data

    assert np.all(np.isin(y_train, (-1, 1))), 'Targets must be ±1'
    assert x_train.shape[0] == y_train.shape[0], 'x_train and y_train have inconsistent sizes'

    N = x_train.shape[0]
    indices = np.arange(N)
    if balance_classes:
        c1_indices = indices[y_train == -1]
        c2_indices = indices[y_train == 1]
        c1_size = M // 2
        c2_size = M - c1_size
        assert (
            c1_size <= c1_indices.shape[0] and c2_size <= c2_indices.shape[0]
        ), 'Not enough samples to balance classes'

        index_sets = []
        while len(index_sets) < S:
            c1_idx = np.random.choice(c1_indices, c1_size, replace=False)
            c2_idx = np.random.choice(c2_indices, c2_size, replace=False)
            idx = np.hstack((c1_idx, c2_idx))
            np.random.shuffle(idx)
            index_sets.append(idx)
    else:
        disjoint = S * M <= N
        if disjoint:
            np.random.shuffle(indices)
            index_sets = [indices[i : i + M] for n, i in enumerate(range(0, N, M)) if n < S]
        else:
            index_sets = [np.random.choice(indices, M, replace=False) for _ in range(S)]

    assert len(index_sets) == S and all(len(idx) == M for idx in index_sets), '`index_sets` computation failed'

    x_train_subsets = [x_train[idx] for idx in index_sets]
    y_train_subsets = [y_train[idx] for idx in index_sets]

    weak_classifiers = []
    failed_anneals = 0
    for x, y in tqdm(zip(x_train_subsets, y_train_subsets), total=S):
        try:
            qsvm = QSVM(**qsvm_params)
            qsvm.fit(x, y)
        except:
            failed_anneals += 1
            print(f'Annealing failed. Sampling with steepest descent for the {failed_anneals}th time.')
            params = deepcopy(qsvm_params)
            params['sampler'] = 'steepest_descent'
            qsvm = QSVM(**params)
            qsvm.fit(x, y)
        weak_classifiers.append(qsvm)

    if failed_anneals:
        print(f'Failed anneals: {failed_anneals}')

    if save is not None:
        with open(save, 'wb') as f:
            dill.dump((x_train_subsets, y_train_subsets, qsvm_params, weak_classifiers), f)

    return x_train_subsets, y_train_subsets, qsvm_params, weak_classifiers


def run_ensembles():
    verbose = True
    visualize = False
    features = ['z', 'normal_variation', 'height_variation', 'log_intensity']

    n_train_samples = 10_000
    n_valid_samples = 100_000

    # _, train_x, train_y, valid_x, valid_y, _, _, _, _ = get_data(
    #     n_train_samples, n_valid_samples, features, verbose, visualize, DATASET, WORKING_DIR
    # )
    with open(ENSEMBLE_LOG_DIR / f'dataset_{DATASET}_ts={n_train_samples}_vs={n_valid_samples}.pkl', 'rb') as f:
        train_x, train_y, valid_x, valid_y = dill.load(f)

    n_train_samples = 5_000
    train_x = train_x[:n_train_samples]
    train_y = train_y[:n_train_samples]

    common_qsvm_params = dict(
        kernel='rbf',
        sampler='qa_clique',
        num_reads=1_000,
        normalize=True,
        hybrid_time_limit=3,
        threshold=0,
        threshold_strategy='relative',
        optimize_memory=False,
        dwave_api_token=None,
        fail_to_classical=True,
    )
    if DATASET == 'kits':
        qsvm_params = dict(B=2, P=1, K=4, zeta=0.0, gamma=0.1778279410038923)
    elif DATASET == 'downtown':
        qsvm_params = dict(B=2, P=2, K=4, zeta=0.4, gamma=1.0)
    elif DATASET == 'ptgrey':
        qsvm_params = dict(B=2, P=2, K=4, zeta=1.2, gamma=1.7782794100389228)
    qsvm_params |= common_qsvm_params

    M = 44  # selected such that num_qubo_elements = 4 * 44 = 176 <= 177 = max_clique_size
    S = n_train_samples // M

    x_train_subsets, y_train_subsets, qsvm_params, weak_classifiers = train_weak_qsvms(
        qsvm_params,
        train_x,
        train_y,
        S=S,
        M=M,
        balance_classes=True,
        # save=ENSEMBLE_LOG_DIR / f'weak_classifiers_{DATASET}_ts={n_train_samples}_{S=}_{M=}.pkl',
        load=ENSEMBLE_LOG_DIR / f'weak_classifiers_{DATASET}_ts={n_train_samples}_{S=}_{M=}.pkl',
    )

    ###################################################################################################################
    # QBoost ##########################################################################################################
    ###################################################################################################################

    qboost_sampler = 'hybrid'
    hybrid_time_limit = 6
    qboost_common_params = dict(
        weak_classifiers=weak_classifiers,
        # lbda=(0.0, 2.1, 0.1),
        lbda=(0.0, 0.251, 0.05),
        num_reads=1_000,
        sampler=qboost_sampler,
        hybrid_time_limit=hybrid_time_limit,
        dwave_api_token=None,
        fail_to_classical=True,
    )
    if DATASET == 'kits':
        qboost_params = dict(B=2, P=3, K=5)
    elif DATASET == 'downtown':
        qboost_params = dict(B=2, P=1, K=8)
    elif DATASET == 'ptgrey':
        qboost_params = dict(B=2, P=2, K=7)
    qboost_params |= qboost_common_params

    # qboost = QBoost(**qboost_params)
    # qboost.fit(train_x, train_y)
    # with open(
    #     ENSEMBLE_LOG_DIR
    #     / f'qboost_{DATASET}_ts={n_train_samples}_vs=None_sampler={qboost_sampler}_htl={hybrid_time_limit}.pkl',
    #     'wb',
    # ) as f:
    #     dill.dump(qboost, f)

    # qboost_valid = QBoost(**qboost_params)
    # qboost_valid.fit(train_x, train_y, valid_x[:n_train_samples], valid_y[:n_train_samples])
    # with open(
    #     ENSEMBLE_LOG_DIR
    #     / f'qboost_{DATASET}_ts={n_train_samples}_vs={n_train_samples}_sampler={qboost_sampler}_htl={hybrid_time_limit}.pkl',
    #     'wb',
    # ) as f:
    #     dill.dump(qboost_valid, f)

    with open(
        ENSEMBLE_LOG_DIR
        / f'qboost_{DATASET}_ts={n_train_samples}_vs=None_sampler={qboost_sampler}_htl={hybrid_time_limit}.pkl',
        'rb',
    ) as f:
        qboost = dill.load(f)
    with open(
        ENSEMBLE_LOG_DIR
        / f'qboost_{DATASET}_ts={n_train_samples}_vs={n_train_samples}_sampler={qboost_sampler}_htl={hybrid_time_limit}.pkl',
        'rb',
    ) as f:
        qboost_valid = dill.load(f)

    preds = qboost.predict(valid_x)
    preds_valid = qboost_valid.predict(valid_x)
    cm = confusion_matrix(preds, valid_y)
    cm_valid = confusion_matrix(preds_valid, valid_y)
    mcc = matthews_corrcoef(cm)
    mcc_valid = matthews_corrcoef(cm_valid)
    f1 = f1_score(cm)
    f1_valid = f1_score(cm_valid)
    acc = accuracy(cm)
    acc_valid = accuracy(cm_valid)

    print(
        f'QBoost:\n\tcm = {cm.tolist()}\n\t{mcc = :.3f} ({mcc_valid:.3f})\n\t{f1 = :.3f} ({f1_valid:.3f})\n\t{acc = :.2%} ({acc_valid:.2%})'
    )

    ###################################################################################################################
    # AdaBoost ########################################################################################################
    ###################################################################################################################

    adaboost_common_params = dict(weak_classifiers=weak_classifiers)
    if DATASET == 'kits':
        adabost_params = dict(n_estimators=64)
    elif DATASET == 'downtown':
        adabost_params = dict(n_estimators=22)
    elif DATASET == 'ptgrey':
        adabost_params = dict(n_estimators=30)
    adabost_params |= adaboost_common_params

    # adaboost = AdaBoost(**adabost_params)
    # adaboost.fit(train_x, train_y)

    # with open(ENSEMBLE_LOG_DIR / f'adaboost_{DATASET}_ts={n_train_samples}.pkl', 'wb') as f:
    #     dill.dump(adaboost, f)

    with open(ENSEMBLE_LOG_DIR / f'adaboost_{DATASET}_ts={n_train_samples}.pkl', 'rb') as f:
        adaboost = dill.load(f)

    preds = adaboost.predict(valid_x)
    cm = confusion_matrix(preds, valid_y)
    mcc = matthews_corrcoef(cm)
    f1 = f1_score(cm)
    acc = accuracy(cm)

    print(f'AdaBoost:\n\tcm = {cm.tolist()}\n\t{mcc = :.3f}\n\t{f1 = :.3f}\n\t{acc = :.2%}')

    ###################################################################################################################
    # Softmax QSVM ####################################################################################################
    ###################################################################################################################

    softmax_qsvm_common_params = dict(weak_classifiers=weak_classifiers, multiplier=10.0)
    softmax_params = {}
    softmax_params |= softmax_qsvm_common_params

    softmax_qsvm = SoftmaxQSVM(**softmax_params)
    softmax_qsvm.fit(train_x, train_y)

    with open(ENSEMBLE_LOG_DIR / f'softmax_qsvm_{DATASET}_ts={n_train_samples}.pkl', 'wb') as f:
        dill.dump(softmax_qsvm, f)

    with open(ENSEMBLE_LOG_DIR / f'softmax_qsvm_{DATASET}_ts={n_train_samples}.pkl', 'rb') as f:
        softmax_qsvm = dill.load(f)

    preds = softmax_qsvm.predict(valid_x)
    cm = confusion_matrix(preds, valid_y)
    mcc = matthews_corrcoef(cm)
    f1 = f1_score(cm)
    acc = accuracy(cm)

    print(f'Softmax QSVM:\n\tcm = {cm.tolist()}\n\t{mcc = :.3f}\n\t{f1 = :.3f}\n\t{acc = :.2%}')


if __name__ == '__main__':
    DATASET = sys.argv[1] if len(sys.argv) > 1 else None
    assert DATASET is None or DATASET in ('kits', 'downtown', 'ptgrey')

    WORKING_DIR = Path(__file__).parent
    ENSEMBLE_LOG_DIR = WORKING_DIR / 'logs' / 'ensemble_logs'

    run_ensembles()
