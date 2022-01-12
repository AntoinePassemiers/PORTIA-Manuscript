# -*- coding: utf-8 -*-
# grn.py
# author: Antoine Passemiers

import numpy as np


class GRN:

    def __init__(self, A, tf_idx):
        self.A = np.asarray(A, dtype=float)
        assert self.A.shape[0] == self.A.shape[1]
        self.tf_idx = np.asarray(tf_idx, dtype=int)

    def add_negatives_inplace(self):
        mask = np.any(self.A == 1, axis=1)
        self.A[mask, :] = np.nan_to_num(self.A[mask, :], nan=0)

    def as_symmetric(self):
        A = np.copy(self.A)

        # If one of the 2 directed edges is 1, then the undirected edge is 1
        mask = np.isnan(self.A)
        A[mask] = 0
        A = np.maximum(A, A.T)

        # If both directed edges are unknown, then the undirected edge is unknown
        mask = np.logical_and(mask, mask.T)
        A[mask] = np.nan

        # Create a new GRN from the adjacency matrix and TF list
        return GRN(A, self.tf_idx)

    def get_mask(self):
        mask = np.zeros(self.shape, dtype=bool)
        mask[self.tf_idx, :] = 1
        np.fill_diagonal(mask, 0)
        mask[np.isnan(self.A)] = 0
        return mask

    def binarize(self, n_top):

        # Missing values will be considered last (-inf is less than anything)
        # or won't be considered at all if n_top is not too big
        A = np.copy(self.A)
        np.nan_to_num(A, nan=-np.inf, copy=False)

        # Values associated to non-regulating genes will be considered last
        # or won't be considered at all if n_top is not too big
        mask = np.zeros(A.shape[0], dtype=bool)
        mask[self.tf_idx] = 1
        A[~mask, :] = -np.inf

        # Create an adjacency matrix from the `n_top` top values
        idx = np.argsort(A, axis=None)[-n_top:]
        idx = np.unravel_index(idx, self.shape)
        A = np.zeros(self.shape, dtype=np.uint8)
        A[idx[0], idx[1]] = 1

        # Create a new GRN from the adjacency matrix and TF list
        return GRN(A, self.tf_idx)

    @staticmethod
    def load_network(filepath, gene_names, tf_idx, from_string=False):
        n_genes = len(gene_names)
        if tf_idx is None:
            tf_idx = np.arange(n_genes)
        A = np.full((n_genes, n_genes), np.nan, dtype=float)
        gene_names = {gene_name: i for i, gene_name in enumerate(gene_names)}
        if from_string:
            lines = filepath.splitlines()
        else:
            with open(filepath, 'r') as f:
                lines = f.readlines()
        for line in lines:
            elements = line.rstrip().split()
            if len(elements) == 3:
                i = gene_names[elements[0]]
                j = gene_names[elements[1]]
                A[i, j] = float(elements[2])
            elif len(elements) == 2:
                i = gene_names[elements[0]]
                j = gene_names[elements[1]]
                A[i, j] = 1
        assert A.shape[0] == A.shape[1]
        np.fill_diagonal(A, 0)
        return GRN(A, tf_idx)

    @staticmethod
    def load_goldstandard(filepath, gene_names, tf_idx=None, from_string=False):
        return GRN.load_network(filepath, gene_names, tf_idx=tf_idx, from_string=from_string)

    def asarray(self):
        return self.A

    def shuffle(self):
        np.random.shuffle(self.A)

    def copy(self):
        A = np.copy(self.A)
        tf_idx = np.copy(self.tf_idx)
        return GRN(A, tf_idx)

    def __getitem__(self, key):
        return self.A[key]

    def __setitem__(self, key, value):
        self.A[key] = value

    @property
    def shape(self):
        return self.A.shape

    @property
    def n_genes(self):
        return self.A.shape[0]

    @property
    def n_edges(self):
        return int(np.nansum(self.A))
