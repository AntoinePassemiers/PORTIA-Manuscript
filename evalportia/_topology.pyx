# -*- coding: utf-8 -*-
# _topology.pyx
# author : Antoine Passemiers
# distutils: language=c
# cython: boundscheck=False
# cython: initializedcheck=False
# cython: nonecheck=False
# cython: overflowcheck=False
# cython: language_level=3

import numpy as np
cimport numpy as cnp
cnp.import_array()

from libc.stdlib cimport *

from evalportia.causal_structure import CausalStructure


cdef cnp.uint8_t TRUE_POSITIVE = int(CausalStructure.TRUE_POSITIVE)
cdef cnp.uint8_t CHAIN = int(CausalStructure.CHAIN)
cdef cnp.uint8_t FORK = int(CausalStructure.FORK)
cdef cnp.uint8_t CHAIN_REVERSED = int(CausalStructure.CHAIN_REVERSED)
cdef cnp.uint8_t COLLIDER = int(CausalStructure.COLLIDER)
cdef cnp.uint8_t UNDIRECTED = int(CausalStructure.UNDIRECTED)
cdef cnp.uint8_t SPURIOUS_CORRELATION = int(CausalStructure.SPURIOUS_CORRELATION)
cdef cnp.uint8_t TRUE_NEGATIVE = int(CausalStructure.TRUE_NEGATIVE)
cdef cnp.uint8_t FALSE_NEGATIVE = int(CausalStructure.FALSE_NEGATIVE)
cdef cnp.float32_t NAN = <cnp.float32_t>np.nan


def _evaluate(_A, _A_pred, _C, _CU, tf_mask):
    """
    Category 0: true positives
    Category 1: chains (indirect causal effects)
    Category 2: forks (no indirect effect, but the variables
        remain d-separated)
    Category 3: colliders, reversed chains and undirected relations
        (causal structures in which variables are not d-separated)
    Category 4: Reversed chains
    Category 5: Undirected links
    Category 6: Spurious correlations (no known causal structure)
    """
    cdef cnp.uint8_t[:, :] C = np.asarray(_C, dtype=np.uint8)
    cdef cnp.uint8_t[:, :] CU = np.asarray(_CU, dtype=np.uint8)
    cdef cnp.uint8_t[:, :] A = np.asarray(_A, dtype=np.uint8)
    cdef cnp.uint8_t[:, :] A_pred = np.asarray(_A_pred, dtype=np.uint8)
    cdef cnp.uint8_t[:, :] T = np.zeros(_A.shape, np.uint8)
    cdef cnp.uint8_t[:] mask = np.asarray(tf_mask, dtype=np.uint8)
    cdef int n_genes = A.shape[0]
    cdef int i, j, k
    cdef bint found

    assert _A.shape == (n_genes, n_genes)
    assert _A_pred.shape == (n_genes, n_genes)
    assert _C.shape == (n_genes, n_genes)
    assert _CU.shape == (n_genes, n_genes)
    assert tf_mask.shape == (n_genes,)

    T[:, :] = TRUE_NEGATIVE

    with nogil:

        for i in range(n_genes):
            if mask[i]:
                for j in range(n_genes):
                    if not A[i, j]:

                        if A_pred[i, j]:  # False positive

                            # Check chains
                            if C[i, j]:
                                T[i, j] = CHAIN
                                continue

                            # Check reversed chains
                            if C[j, i]:
                                T[i, j] = CHAIN_REVERSED
                                continue

                            # Check colliders
                            found = False
                            for k in range(n_genes):
                                if C[i, k] and C[j, k]:
                                    found = True
                                    break
                            if found:
                                T[i, j] = COLLIDER
                                continue

                            # Check forks
                            found = False
                            for k in range(n_genes):
                                if mask[k]:
                                    if C[k, i] and C[k, j]:
                                        found = True
                                        break
                            if found:
                                T[i, j] = FORK
                                continue

                            # Check undirected
                            if CU[i, j]:
                                T[i, j] = UNDIRECTED
                                continue

                            T[i, j] = SPURIOUS_CORRELATION
                            continue

                        else:  # True negative
                            T[i, j] = TRUE_NEGATIVE

                    elif A[i, j]:
                        if A_pred[i, j]:
                            T[i, j] = TRUE_POSITIVE
                        else:
                            T[i, j] = FALSE_NEGATIVE

    return np.asarray(T)
