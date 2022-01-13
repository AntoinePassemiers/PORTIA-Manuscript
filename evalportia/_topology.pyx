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


def _all_connected(A, _tf_mask):
    """Naive algorithm for finding all pairs of connected vertices.

    Note: possible improvement, compute accessibility matrix
    (e.g. with Floyd-Warshall algorithm)
    """
    cdef cnp.uint8_t[:, :] C = np.copy(np.asarray(A, dtype=np.uint8))
    cdef cnp.uint8_t[:] mask = np.asarray(_tf_mask, dtype=np.uint8)
    cdef int i, j, k
    cdef bint found_new_connection = True
    with nogil:
        while found_new_connection:
            found_new_connection = False
            for i in range(C.shape[0]):
                if mask[i]:
                    for j in range(C.shape[1]):
                        if not C[i, j]:
                            for k in range(C.shape[1]):
                                if C[i, k] and C[k, j]:
                                    C[i, j] = 1
                                    found_new_connection = True
                                    break
    return np.asarray(C)


def _evaluate(_A, _A_pred, _C, _CU, tf_mask):
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

                            # Check forks
                            found = False
                            for k in range(n_genes):
                                if C[k, i] and C[k, j]:
                                    found = True
                                    break
                            if found:
                                T[i, j] = FORK
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

                            # Check undirected
                            if CU[i, j]:
                                T[i, j] = UNDIRECTED
                                continue

                            # No explanation could be found to justify the
                            # presence of a false positive -> spurious correlation
                            T[i, j] = SPURIOUS_CORRELATION

                        else:  # True negative
                            T[i, j] = TRUE_NEGATIVE

                    else:  # A[i, j] == True
                        if A_pred[i, j]:  # True positive
                            T[i, j] = TRUE_POSITIVE
                        else:  # False negative
                            T[i, j] = FALSE_NEGATIVE

    return np.asarray(T)
