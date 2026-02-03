from __future__ import annotations
from typing import Tuple
import numpy as np

from server.trainer.ga.base import Individual


class NoCrossover:
    """交叉を行わず、そのままコピー"""
    def __call__(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        c1 = Individual(
            p1.body.copy(),
            p1.connections.copy(),
            p1.label,
            p1.controller_params,
        )
        c2 = Individual(
            p2.body.copy(),
            p2.connections.copy(),
            p2.label,
            p2.controller_params,
        )
        return c1, c2


class RowCrossover:
 

    def __call__(self, p1: Individual, p2: Individual) -> Tuple[Individual, Individual]:
        body1 = p1.body
        body2 = p2.body

        if body1.shape != body2.shape or body1.shape[0] <= 1:
            return NoCrossover()(p1, p2)

        h = body1.shape[0]
        cut = np.random.randint(1, h)

        child1_body = np.vstack([body1[:cut], body2[cut:]])
        child2_body = np.vstack([body2[:cut], body1[cut:]])

        c1 = Individual(
            child1_body,
            p1.connections.copy(),
            p1.label,
            p1.controller_params,
        )
        c2 = Individual(
            child2_body,
            p2.connections.copy(),
            p2.label,
            p2.controller_params,
        )

        return c1, c2
