# -*- coding: utf-8 -*-
# causal_structure.py
# author: Antoine Passemiers

import enum

import numpy as np


class CausalStructure(enum.IntEnum):

    # True positive regulatory link
    TRUE_POSITIVE = enum.auto()

    # Indirectly regulated genes
    CHAIN = enum.auto()

    # D-connected genes
    FORK = enum.auto()

    # Indirectly regulated genes, with link predicted
    # in the wrong direction
    CHAIN_REVERSED = enum.auto()

    # Collider (d-separated genes)
    COLLIDER = enum.auto()

    # Undirected link (d-separated genes)
    UNDIRECTED = enum.auto()

    # Spurious correlation
    SPURIOUS_CORRELATION = enum.auto()

    # True negative
    TRUE_NEGATIVE = enum.auto()

    # False negative
    FALSE_NEGATIVE = enum.auto()

    @staticmethod
    def array_to_relevance(M):
        R = np.zeros(M.shape, dtype=np.uint8)
        R[M == CausalStructure.TRUE_POSITIVE] = 4
        R[M == CausalStructure.CHAIN] = 3
        R[M == CausalStructure.FORK] = 2
        R[M == CausalStructure.CHAIN_REVERSED] = 1
        R[M == CausalStructure.COLLIDER] = 1
        R[M == CausalStructure.UNDIRECTED] = 1
        R[M == CausalStructure.SPURIOUS_CORRELATION] = 0

        # Special cases
        R[M == CausalStructure.TRUE_NEGATIVE] = 0  # TODO
        R[M == CausalStructure.FALSE_NEGATIVE] = 0
        return R

    @staticmethod
    def to_string(i):
        if i == CausalStructure.TRUE_POSITIVE:
            return 'True positive'
        elif i == CausalStructure.CHAIN:
            return 'Chain'
        elif i == CausalStructure.FORK:
            return 'Fork'
        elif i == CausalStructure.CHAIN_REVERSED:
            return 'Chain (reversed)'
        elif i == CausalStructure.COLLIDER:
            return 'Collider'
        elif i == CausalStructure.UNDIRECTED:
            return 'Undirected'
        elif i == CausalStructure.SPURIOUS_CORRELATION:
            return 'Spurious correlation'
        elif i == CausalStructure.TRUE_NEGATIVE:
            return 'True negative'
        elif i == CausalStructure.FALSE_NEGATIVE:
            return 'False negative'
        else:
            return 'Unknown'
