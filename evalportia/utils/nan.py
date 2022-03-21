# -*- coding: utf-8 -*-
# nan.py
# author: Antoine Passemiers

import numpy as np


def nan_to_min(x):
    x = np.copy(x)
    mask = np.isnan(x)
    if np.sum(~mask) == 0:
        # All predictions are missing: NaNs are replaced with 0
        x[mask] = 0
    else:
        # Randomly shuffle missing gene pairs / gene pairs with lowest score
        x[mask] = np.min(x[~mask]) - 1e-3 * np.random.rand(*x[mask].shape)
    return x
