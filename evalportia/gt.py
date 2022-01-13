# -*- coding: utf-8 -*-
# gt.py
# author: Antoine Passemiers

import os
import sys

import numpy as np

from evalportia.causal_structure import CausalStructure
from evalportia.utils.nan import nan_to_min

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from _topology import _evaluate, _all_connected


def are_nodes_connected(A, tf_mask):
    return _all_connected(A, tf_mask)


def graph_theoretic_evaluation(filepath, G_target, G_pred, tf_mask):
    n_genes = G_target.n_genes
    n_edges = G_target.n_edges

    # Goldstandard adjacency matrix
    A = np.copy(G_target.asarray())

    # Goldstandard undirected (symmetric) adjacency matrix
    AU = G_target.as_symmetric().asarray()

    # Convert predicted scores to binary adjacency matrix
    A_binary_pred = G_pred.binarize(n_edges).asarray()
    assert int(np.sum(A_binary_pred)) == n_edges

    # Fill missing values
    np.nan_to_num(A, nan=0, copy=False)
    np.nan_to_num(AU, nan=0, copy=False)
    A_binary_pred = nan_to_min(A_binary_pred)

    if os.path.exists(filepath):
        data = np.load(filepath)
        C = data['C']
        CU = data['CU']
    else:
        # Find all pairs of connected vertices
        C = are_nodes_connected(A, tf_mask)

        # Find all pairs of connected vertices (undirected edges)
        CU = are_nodes_connected(AU, np.ones(AU.shape[0]))

        # If `i` indirectly regulates `j`, then an undirected regulatory relationship
        # exists between `j` and `i`, and vice versa
        # This step is necessary because we didn't compute node connectivity when tf_mask[i]
        # is False, leading to missing values in `CU`
        CU = np.maximum(CU, CU.T)

        # Save regulatory relationship matrices
        np.savez(filepath, C=C, CU=CU)
    assert np.all(CU >= C)

    # No self-regulation
    np.fill_diagonal(A, 0)
    np.fill_diagonal(C, 0)
    np.fill_diagonal(CU, 0)

    # Categorise predictions based on local causal structures
    results = {'T': _evaluate(A, A_binary_pred, C, CU, tf_mask)}

    # Convert the matrix of causal structures into a relevance matrix
    relevance = CausalStructure.array_to_relevance(results['T'])

    # Filter predictions and keep only meaningful ones:
    # no self-regulation, and no prediction for TFs for which there is no
    # experimental evidence, based on the goldstandard
    mask = G_target.get_mask()
    y_relevance = relevance[mask]
    y_pred = G_pred[mask]

    # Fill missing predictions with the lowest score.
    # For fair comparison of the different GRN inference methods,
    # the same number of predictions should be reported.
    # If a method reports less regulatory links than what is present
    # in the goldstandard, missing values will be consumed until
    # the right number of predictions (`n_edges`) is reached.
    y_pred = nan_to_min(y_pred)

    # Get the indices of the first `n_edges` predictions,
    # sorted by decreasing order of importance
    idx = np.argsort(y_pred)[-n_edges:][::-1]

    # Compute importance weights
    weights = 1. / np.log2(1. + np.arange(1, len(idx) + 1))

    # Discounted Cumulative Gain
    dcg = np.sum(weights * y_relevance[idx])

    # Ideal Discounted Cumulative Gain
    idcg = 4. * np.sum(weights)

    # Normalized Discounted Cumulative Gain (ranges between 0 and 1)
    ndcg = dcg / idcg

    # Store NDCG score
    results['score'] = ndcg
    return results
