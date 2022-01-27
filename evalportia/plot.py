# -*- coding: utf-8 -*-
# utils.py
# author: Antoine Passemiers

import numpy as np
import matplotlib.pyplot as plt


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
