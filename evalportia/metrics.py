# -*- coding: utf-8 -*-
# metrics.py
# author: Antoine Passemiers

import numpy as np
import pandas as pd
import scipy.special
from sklearn.metrics import auc, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.neighbors import KernelDensity

from portia.gt.symmetry import matrix_symmetry
from evalportia.utils.nan import nan_to_min
from evalportia.curves import get_statistics


def compute_auroc(y_target, y_pred):
    if np.sum(y_target) == 0:
        # return 0.5, [], []
        raise NotImplementedError()
    else:
        fpr, tpr, _ = roc_curve(y_target, y_pred)
        return roc_auc_score(y_target, y_pred, average='weighted'), (fpr, tpr)


def compute_auprc(y_target, y_pred):
    if np.sum(y_target) == 0:
        # return 0, [], []
        raise NotImplementedError()
    else:
        eta = np.mean(y_target)
        sample_weight = np.ones(len(y_target))
        sample_weight[y_target == 1] = 0.5 / np.sum(y_target == 1)
        sample_weight[y_target == 0] = 0.5 / np.sum(y_target == 0)
        precision, recall, _ = precision_recall_curve(y_target, y_pred, sample_weight=sample_weight)
        return auc(recall, precision), (precision, recall)


def estimate_distribution(data, n_points=1000, relative_range=5):
    bandwidth = 0.5 * np.std(data)
    kde = KernelDensity(kernel='exponential', bandwidth=bandwidth)
    kde.fit(data.reshape(-1, 1))
    _min = np.min(data)
    _max = np.max(data)
    _range = _max - _min
    new_min = _min - 0.5 * (relative_range - 1) * _range
    new_max = _max + 0.5 * (relative_range - 1) * _range
    x = np.linspace(new_min, new_max, n_points)
    y = kde.score_samples(x.reshape(-1, 1)).flatten()
    y = scipy.special.softmax(y)
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))
    return x, y


def compute_area_distributions(G_target, n_networks=25000, n_top=30000):
    mask = G_target.get_mask()
    G_random = G_target.copy()
    areas = np.empty((2, n_networks))

    for i in range(n_networks):

        """
        in_degree = np.sum(G_target[:, :] == 1, axis=0)
        in_degree = in_degree / np.maximum(1, np.sum(~np.isnan(G_target[:, :]), axis=0))
        G_random = (np.random.rand(*G_target.shape) < in_degree).astype(int)
        """

        y_target = G_target[mask]
        y_pred = G_random[mask]
        np.random.shuffle(y_pred)

        y_pred_noisy = y_pred + 0.01 * np.random.rand(*y_pred.shape)

        # Get the indices of the first `n_edges` predictions,
        # sorted by decreasing order of importance
        idx = np.argsort(y_pred_noisy)[-n_top:]
        y_target = y_target[idx]
        y_pred_noisy = y_pred_noisy[idx]

        auprc, auroc, _, _, _, _ = get_statistics(y_target[::-1], y_pred_noisy[::-1])
        areas[0, i] = auroc
        areas[1, i] = auprc

    pdf_data = {}
    x, y = estimate_distribution(areas[0, :])
    pdf_data['x_auroc'] = x
    pdf_data['y_auroc'] = y
    pdf_data['aurocs'] = areas[0, :]
    x, y = estimate_distribution(areas[1, :])
    pdf_data['x_auprc'] = x
    pdf_data['y_auprc'] = y
    pdf_data['auprcs'] = areas[1, :]

    return pdf_data


def compute_p_value(value, x, y):
    x = np.squeeze(x)
    y = np.squeeze(y)
    last_value = x[-1] + (x[-1] - x[-2])
    dx = np.diff(x, append=last_value)
    mask = (x >= value)
    p_value = np.sum(y[mask] * dx[mask])
    total = np.sum(y * dx)
    if total > 0:
        p_value /= total
    return p_value


def compute_metrics(G_target, G_pred, pdf_data, n_top=30000):

    # Filter predictions and keep only meaningful ones:
    # no self-regulation, and no prediction for TF for which there is no
    # experimental evidence, based on the goldstandard
    mask = G_target.get_mask()
    y_target = G_target[mask]
    y_pred = G_pred[mask]

    # Fill missing predictions with the lowest score (+ some random noise
    # in order to shuffle the missing gene pairs).
    # For fair comparison of the different GRN inference methods,
    # the same number of predictions should be reported.
    # If a method reports less regulatory links than what is present
    # in the goldstandard, missing values will be consumed until
    # the right number of predictions (`n_edges`) is reached.
    y_pred = nan_to_min(y_pred)

    # Get the indices of the first `n_edges` predictions,
    # sorted by decreasing order of importance
    idx = np.argsort(y_pred)[-n_top:]
    y_target = y_target[idx]
    y_pred = y_pred[idx]

    assert not np.any(np.isnan(y_target))
    assert len(np.unique(y_target) == 2)

    auprc, auroc, precision, recall, tpr, fpr = get_statistics(y_target[::-1], y_pred[::-1])
    roc_curve = (tpr, fpr)
    prc_curve = (precision, recall)

    #auroc, roc_curve = compute_auroc(y_target, y_pred)
    #auprc, prc_curve = compute_auprc(y_target, y_pred)
    #p_value_auroc = np.mean(auroc < pdf_data['aurocs'])
    #p_value_auprc = np.mean(auprc < pdf_data['auprcs'])
    p_value_auroc = compute_p_value(auroc, pdf_data['x_auroc'], pdf_data['y_auroc'])
    p_value_auprc = compute_p_value(auprc, pdf_data['x_auprc'], pdf_data['y_auprc'])

    LOG_ZERO = -300
    if p_value_auprc == 0:
        log_p_value_auprc = LOG_ZERO
    else:
        log_p_value_auprc = np.log10(p_value_auprc)
    if p_value_auroc == 0:
        log_p_value_auroc = LOG_ZERO
    else:
        log_p_value_auroc = np.log10(p_value_auroc)
        
    score = -0.5 * (log_p_value_auroc + log_p_value_auprc)
    return {
        'auprc': float(auprc),
        'auroc': float(auroc),
        'auprc-p-value': float(p_value_auprc),
        'auroc-p-value': float(p_value_auroc),
        'score': float(score),
        'symmetry': float(matrix_symmetry(G_pred)),
        'target-symmetry': float(matrix_symmetry(G_target)),

        'prc-curve': list(prc_curve),
        'roc-curve': list(roc_curve),
        'y-pred': list(y_pred),
        'y-target': list(y_target)
    }


def score_dream_prediction(gs_filepath, pred_filepath, pdf_data, use_test=False):
    """Note: inspired by:
    https://github.com/dreamtools/dreamtools/blob/master/dreamtools/dream3/D3C4/scoring.py
    https://github.com/dreamtools/dreamtools/blob/master/dreamtools/dream4/D4C2/scoring.py
    """
    import dreamtools
    d3d4roc = dreamtools.core.rocs.D3D4ROC()

    def _load_network(filename):
        df = pd.read_csv(filename, header=None, sep='[ \t]', engine='python')
        df[0] = df[0].apply(lambda x: x.replace('g', '').replace('G', ''))
        df[1] = df[1].apply(lambda x: x.replace('g', '').replace('G', ''))
        return df.astype(float)

    gold_data = _load_network(gs_filepath)
    test_data = _load_network(pred_filepath)

    newtest = pd.merge(test_data, gold_data, how='inner', on=[0, 1])
    if use_test:
        test = list(newtest['2_x'])
    else:
        test = test_data
    gold_index = list(newtest['2_y'])
    auprc, auroc, _, _, _, _ = d3d4roc.get_statistics(gold_data, test, gold_index)

    def _probability(X, Y, x):
        dx = X[2] - X[1]
        P = sum(Y[X >= x] * dx)
        return P

    p_auroc = _probability(pdf_data['x_auroc'].flatten(), pdf_data['y_auroc'].flatten(), auroc)
    p_aupr = _probability(pdf_data['x_auprc'].flatten(), pdf_data['y_auprc'].flatten(), auprc)

    log_p_auroc = -300 if (p_auroc == 0) else np.log10(p_auroc)
    log_p_auprc = -300 if (p_aupr == 0) else np.log10(p_aupr)

    score = -0.5 * (log_p_auroc + log_p_auprc)

    return {'auprc': auprc, 'auroc': auroc, 'score': score}
