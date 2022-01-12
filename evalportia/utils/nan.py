# -*- coding: utf-8 -*-
# nan.py
# author: Antoine Passemiers

import numpy as np


def nan_to_min(x):
    x = np.copy(x)
    mask = np.isnan(x)
    if np.sum(~mask) == 0:
        x[mask] = 0
    else:
        x[mask] = np.min(x[~mask])
    return x
