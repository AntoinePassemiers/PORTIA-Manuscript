# -*- coding: utf-8 -*-
# utils.py
# author: Antoine Passemiers

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from sklearn.metrics import matthews_corrcoef

from evalportia.causal_structure import CausalStructure


def plot_matrix_symmetry(values, method_names, title='', network_names=None, color_idx=None):


    plt.rcParams['figure.figsize'] = [8, 4]

    values = np.asarray(values)
    n_networks = values.shape[1]
    n_methods = len(method_names)
    colors = ['palevioletred', 'orchid', 'indigo', 'royalblue', 'turquoise']
    width = 0.8 / n_networks

    if network_names is None:
        network_names = [f'Net {i + 1}' for i in range(n_networks)]

    if color_idx is None:
        color_idx = []
        for i in range(len(network_names)):
            color_idx.append(i % len(colors))

    xs = np.arange(n_methods)
    ax = plt.subplot(111)
    for i in range(n_networks):
        ys = values[:, i]
        color = colors[color_idx[i]]
        net_name = network_names[i]
        if net_name is not None:
            ax.bar(xs + width * (i + 1) - 0.4, ys, width, label=net_name, color=color)
        else:
            ax.bar(xs + width * (i + 1) - 0.4, ys, width, color=color)
      
    plt.xticks(xs, method_names, rotation=45)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.ylabel('Matrix symmetry')
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_fp_types(ax, G_target, G_pred, T, n_pred=300):
    n_pred = int(n_pred)
    n_links = G_target.n_edges

    plt.rcParams['figure.figsize'] = [12, 4]

    A = G_target.asarray()
    S = G_pred.asarray()

    mask = G_target.get_mask()
    _as = A[mask]
    ys = np.nan_to_num(S[mask], nan=0)
    types = T[mask]

    idx = np.argsort(ys)[::-1][:n_pred]
    types = types[idx]
    ys = ys[idx]
    _as = _as[idx]
    assert not np.any(np.isnan(types))
    assert not np.any(np.isnan(ys))
    assert not np.any(np.isnan(_as))
    for i in range(len(ys)):
        if not _as[i]:
            if types[i] > 0:
                if types[i] == CausalStructure.CHAIN:
                    color = 'green'
                elif types[i] == CausalStructure.FORK:
                    color = 'orange'
                elif types[i] == CausalStructure.CHAIN_REVERSED:
                    color = 'red'
                elif types[i] == CausalStructure.COLLIDER:
                    color = 'cyan'
                elif types[i] == CausalStructure.UNDIRECTED:
                    color = 'purple'
                elif types[i] == CausalStructure.SPURIOUS_CORRELATION:
                    color = 'black'
                else:
                    color = 'none'
                if color != 'none':
                    ax.bar(i, ys[i], width=1, color=color)
    ax.set_xlim(0, n_pred)
    plt.axvline(x=n_links, linewidth=1, linestyle='--', color='gray', alpha=0.5)
    plt.tick_params(labelleft=False)
    ax.set_yscale('log')
    ax.set(yticklabels=[" "])
    ax.axes.yaxis.set_ticklabels([" "])
    ax.axes.yaxis.set_visible(False)
