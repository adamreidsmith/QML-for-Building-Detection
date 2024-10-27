import sys
from pathlib import Path

import dill
import numpy as np
from sklearn.svm import SVC

from qsvm import QSVM
from quantum_kernels import (
    Kernel,
    data_reuploading_feature_map,
)
from utils import accuracy, confusion_matrix, matthews_corrcoef, f1_score


def run_large():
    n_train_samples = 10_000
    n_valid_samples = 100_000

    with open(DATA_DIR / f'dataset_{DATASET}_ts={n_train_samples}_vs={n_valid_samples}.pkl', 'rb') as f:
        train_x, train_y, valid_x, valid_y = dill.load(f)

    n_train_samples = 5_000
    train_x = train_x[:n_train_samples]
    train_y = train_y[:n_train_samples]

    train_mean = np.mean(train_x, axis=0)
    train_std = np.std(train_x, axis=0)
    train_x_normalized = (train_x - train_mean) / train_std
    valid_x_normalized = (valid_x - train_mean) / train_std

    ###################################################################################################################
    # SVM #############################################################################################################
    ###################################################################################################################

    svm_common_params = dict(kernel='rbf', class_weight='balanced')
    if DATASET == 'kits':
        params = dict(C=0.1778279410038923, gamma=1.0)
    elif DATASET == 'downtown':
        params = dict(C=5.623413251903491, gamma=1.0)
    elif DATASET == 'ptgrey':
        params = dict(C=0.1, gamma=5.623413251903491)
    params |= svm_common_params

    svm = SVC(**params)
    svm.fit(train_x_normalized, train_y)

    with open(LOG_DIR / f'svm_{DATASET}_ts={n_train_samples}.pkl', 'wb') as f:
        dill.dump(svm, f)

    with open(LOG_DIR / f'svm_{DATASET}_ts={n_train_samples}.pkl', 'rb') as f:
        svm = dill.load(f)

    preds = svm.predict(valid_x_normalized)
    cm = confusion_matrix(preds, valid_y)
    mcc = matthews_corrcoef(cm)
    f1 = f1_score(cm)
    acc = accuracy(cm)

    print(f'SVM:\n\tcm = {cm.tolist()}\n\t{mcc = :.3f}\n\t{f1 = :.3f}\n\t{acc = :.2%}')
    del svm

    ###################################################################################################################
    # QSVM ############################################################################################################
    ###################################################################################################################

    qsvm_sampler = 'hybrid'
    qsvm_hybrid_time_limit = 240
    qsvm_common_params = dict(
        kernel='rbf',
        sampler=qsvm_sampler,
        hybrid_time_limit=qsvm_hybrid_time_limit,
        num_reads=1_000,
        normalize=True,
        optimize_memory=False,
        dwave_api_token=None,
        fail_to_classical=False,
    )
    if DATASET == 'kits':
        params = dict(
            B=2, P=1, K=4, zeta=0.0, gamma=0.1778279410038923, threshold=0.0001, threshold_strategy='relative'
        )
    elif DATASET == 'downtown':
        params = dict(B=2, P=2, K=4, zeta=0.4, gamma=1.0, threshold=0.0001, threshold_strategy='relative')
    elif DATASET == 'ptgrey':
        params = dict(
            B=2, P=2, K=4, zeta=1.2, gamma=1.7782794100389228, threshold=0.0001, threshold_strategy='relative'
        )
    params |= qsvm_common_params

    qsvm = QSVM(**params)
    qsvm.fit(train_x, train_y)

    with open(
        LOG_DIR / f'qsvm_2_{DATASET}_ts={n_train_samples}_sampler={qsvm_sampler}_htl={qsvm_hybrid_time_limit}.pkl',
        'wb',
    ) as f:
        dill.dump(qsvm, f)

    with open(
        LOG_DIR / f'qsvm_{DATASET}_ts={n_train_samples}_sampler={qsvm_sampler}_htl={qsvm_hybrid_time_limit}.pkl',
        'rb',
    ) as f:
        qsvm = dill.load(f)

    preds = qsvm.predict(valid_x)
    cm = confusion_matrix(preds, valid_y)
    mcc = matthews_corrcoef(cm)
    f1 = f1_score(cm)
    acc = accuracy(cm)

    print(f'QSVM:\n\tcm = {cm.tolist()}\n\t{mcc = :.3f}\n\t{f1 = :.3f}\n\t{acc = :.2%}')
    del qsvm

    ###################################################################################################################
    # SVM w/ Quantum Kernel ###########################################################################################
    ###################################################################################################################

    kernel_svm_common_params = dict(class_weight='balanced')
    if DATASET == 'kits':
        kernel = Kernel(
            data_reuploading_feature_map(num_features=4, reps=1, entanglement='full'),
            'data_reuploading_feature_map(num_features=4, reps=1, entanglement=\'full\')',
        )
        params = dict(C=1.7782794100389228, kernel=kernel)
    elif DATASET == 'downtown':
        kernel = Kernel(
            data_reuploading_feature_map(num_features=4, reps=1, entanglement='full'),
            'data_reuploading_feature_map(num_features=4, reps=1, entanglement=\'full\')',
        )
        params = dict(C=10, kernel=kernel)
    elif DATASET == 'ptgrey':
        kernel = Kernel(
            data_reuploading_feature_map(num_features=4, reps=1, entanglement='linear'),
            'data_reuploading_feature_map(num_features=4, reps=1, entanglement=\'linear\')',
        )
        params = dict(C=10, kernel=kernel)
    params |= kernel_svm_common_params

    kernel_svm = SVC(**params)
    kernel_svm.fit(train_x_normalized, train_y)

    with open(LOG_DIR / f'kernel_svm_{DATASET}_ts={n_train_samples}.pkl', 'wb') as f:
        dill.dump(kernel_svm, f)

    with open(LOG_DIR / f'kernel_svm_{DATASET}_ts={n_train_samples}.pkl', 'rb') as f:
        kernel_svm = dill.load(f)

    preds = kernel_svm.predict(valid_x_normalized)
    cm = confusion_matrix(preds, valid_y)
    mcc = matthews_corrcoef(cm)
    f1 = f1_score(cm)
    acc = accuracy(cm)

    print(f'Quantum Kernel SVM:\n\tcm = {cm.tolist()}\n\t{mcc = :.3f}\n\t{f1 = :.3f}\n\t{acc = :.2%}')
    del kernel_svm

    ###################################################################################################################
    # QSVM w/ Quantum Kernels #########################################################################################
    ###################################################################################################################

    kernel_qsvm_sampler = 'hybrid'
    kernel_qsvm_hybrid_time_limit = 120
    kernel_qsvm_common_params = dict(
        sampler=kernel_qsvm_sampler,
        hybrid_time_limit=kernel_qsvm_hybrid_time_limit,
        num_reads=1_000,
        normalize=True,
        threshold=0,
        threshold_strategy='absolute',
        optimize_memory=True,
        dwave_api_token=None,
        fail_to_classical=False,
    )
    if DATASET == 'kits':
        kernel = Kernel(
            data_reuploading_feature_map(num_features=4, reps=1, entanglement='full'),
            'data_reuploading_feature_map(num_features=4, reps=1, entanglement=\'full\')',
        )
        params = dict(B=2, P=1, K=3, zeta=0.8, kernel=kernel)
    elif DATASET == 'downtown':
        kernel = Kernel(
            data_reuploading_feature_map(num_features=4, reps=1, entanglement='linear'),
            'data_reuploading_feature_map(num_features=4, reps=1, entanglement=\'full\')',
        )
        params = dict(B=2, P=1, K=3, zeta=0.4, kernel=kernel)
    elif DATASET == 'ptgrey':
        kernel = Kernel(
            data_reuploading_feature_map(num_features=4, reps=1, entanglement='full'),
            'data_reuploading_feature_map(num_features=4, reps=1, entanglement=\'full\')',
        )
        params = dict(B=2, P=1, K=3, zeta=0.4, kernel=kernel)
    params |= kernel_qsvm_common_params

    kernel_qsvm = QSVM(**params)
    kernel_qsvm.fit(train_x, train_y)

    with open(
        LOG_DIR
        / f'kernel_qsvm_{DATASET}_ts={n_train_samples}_sampler={kernel_qsvm_sampler}_htl={kernel_qsvm_hybrid_time_limit}.pkl',
        'wb',
    ) as f:
        dill.dump(kernel_qsvm, f)

    with open(
        LOG_DIR
        / f'kernel_qsvm_{DATASET}_ts={n_train_samples}_sampler={kernel_qsvm_sampler}_htl={kernel_qsvm_hybrid_time_limit}.pkl',
        'rb',
    ) as f:
        kernel_qsvm = dill.load(f)

    preds = kernel_qsvm.predict(valid_x)
    cm = confusion_matrix(preds, valid_y)
    mcc = matthews_corrcoef(cm)
    f1 = f1_score(cm)
    acc = accuracy(cm)

    print(f'Quantum Kernel QSVM:\n\tcm = {cm.tolist()}\n\t{mcc = :.3f}\n\t{f1 = :.3f}\n\t{acc = :.2%}')
    del kernel_qsvm


if __name__ == '__main__':
    DATASET = sys.argv[1] if len(sys.argv) > 1 else None
    assert DATASET is None or DATASET in ('kits', 'downtown', 'ptgrey')

    WORKING_DIR = Path(__file__).parent
    DATA_DIR = WORKING_DIR / 'logs' / 'ensemble_logs'
    LOG_DIR = WORKING_DIR / 'logs' / f'large_logs_{DATASET}'

    run_large()
