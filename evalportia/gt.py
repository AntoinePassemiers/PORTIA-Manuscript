# -*- coding: utf-8 -*-
# gt.py
# author: Antoine Passemiers

import os
import sys

import numpy as np
import igraph

from evalportia.causal_structure import CausalStructure
from evalportia.utils.nan import nan_to_min

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(ROOT)
import pyximport
pyximport.install(setup_args={'include_dirs': np.get_include()})
from _topology import _evaluate


def graph_theoretic_evaluation(filepath, G_target, G_pred, tf_mask):
    n_genes = G_target.n_genes
    n_edges = G_target.n_edges

    # Goldstandard adjacency matrix
    A = G_target.asarray()

    # Goldstandard undirected (symmetric) adjacency matrix
    AU = G_target.as_symmetric().asarray()

    # Convert predicted scores to binary adjacency matrix
    A_binary_pred = G_pred.binarize(n_edges).asarray()

    if os.path.exists(filepath):
        data = np.load(filepath)
        C = data['C']
        CU = data['CU']
    else:
        G1 = igraph.Graph.Adjacency(A.astype(bool).tolist())
        G1U = igraph.Graph.Adjacency(AU.astype(bool).tolist())
        C = np.copy(A).astype(np.uint8)
        CU = np.copy(AU).astype(np.uint8)
        for i in range(n_genes):
            if tf_mask[i]: # C[i, j] cannot be True if `i` is not a TF
                for j in range(n_genes):
                    if j == i:
                        # A gene cannot regulate itself
                        continue
                    if not A[i, j]:
                        # If `i` does not directly regulate `j`, then check whether
                        # it indirectly regulates it
                        C[i, j] = G1.vertex_connectivity(source=i, target=j, checks=False)
                for j in range(i):
                    if not AU[i, j]:
                        # If `i` does not directly regulate `j`, then check whether
                        # it indirectly regulates it (undirected regulation)
                        CU[i, j] = G1U.vertex_connectivity(source=i, target=j, checks=False)
        # If `i` indirectly regulates `j`, then an undirected regulatory relationship
        # exists between `j` and `i`, and vice versa
        # This step is necessary because we didn't compute node connectivity when tf_mask[i]
        # is False, leading to missing values in `CU`
        CU = np.maximum(CU, CU.T)

        # Save regulatory relationship matrices (long running times)
        np.savez(filepath, C=C, CU=CU)
    results = {'T': _evaluate(A, A_binary_pred, C, CU, tf_mask)}

    # Convert the matrix of causal structures into a relevance matrix
    relevance = CausalStructure.array_to_relevance(results['T'])

    # Filter predictions and keep only meaningful ones:
    # no self-regulation, and no prediction for TF for which there is no
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
    weights = np.log2(1. + np.arange(1, len(idx) + 1))

    # Discounted Cumulative Gain
    dcg = np.sum(y_relevance[idx] / weights)

    # Ideal Discounted Cumulative Gain
    idcg = np.sum(4. / weights)

    # Normalized Discounted Cumulative Gain (ranges between 0 and 1)
    ndcg = dcg / idcg

    # Store NDCG score
    results['score'] = ndcg
    return results
